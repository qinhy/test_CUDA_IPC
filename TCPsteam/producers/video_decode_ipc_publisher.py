#!/usr/bin/env python3
from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import PyNvVideoCodec as nvc

from ipc_common.cuda_ipc_ring import CudaIpcRingPublisher
from ipc_common.stream_specs import STREAMS, StreamSpec, parse_stream_names
from ipc_common.torch_cupy import decoded_tensor_to_chw_uint8, preprocess_for_ai


class SocketFeeder:
    """
    Callback feeder for nvc.CreateDemuxer(callback).

    PyNvVideoCodec gives us a pre-allocated demuxer buffer.
    We fill it from the TCP socket and return the number of bytes copied.
    Returning 0 means EOF/reconnect.
    """

    def __init__(self, ip: str, port: int, stop_event: threading.Event):
        self.ip = ip
        self.port = int(port)
        self.stop_event = stop_event
        self.sock: socket.socket | None = None

    def connect(self) -> None:
        while not self.stop_event.is_set():
            try:
                print(f"[tcp:{self.port}] connecting to {self.ip}:{self.port}", flush=True)
                self.sock = socket.create_connection((self.ip, self.port), timeout=5)
                self.sock.settimeout(1.0)
                try:
                    self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except Exception:
                    pass
                print(f"[tcp:{self.port}] connected", flush=True)
                return
            except Exception as e:
                print(f"[tcp:{self.port}] connect failed: {e}", flush=True)
                time.sleep(1.0)

        raise RuntimeError("stopped before connecting")

    def feed_chunk(self, demuxer_buffer) -> int:
        if self.stop_event.is_set():
            return 0

        if self.sock is None:
            self.connect()

        capacity = len(demuxer_buffer)

        while not self.stop_event.is_set():
            try:
                assert self.sock is not None
                data = self.sock.recv(capacity)
                if not data:
                    return 0

                n = len(data)
                demuxer_buffer[:n] = data
                return n

            except socket.timeout:
                continue

            except Exception as e:
                print(f"[tcp:{self.port}] socket read error: {e}", flush=True)
                return 0

        return 0

    def close(self) -> None:
        try:
            if self.sock is not None:
                self.sock.close()
        except Exception:
            pass
        self.sock = None


class DummyAI(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return {
            "shape": tuple(x.shape),
            "device": str(x.device),
            "mean": float(x.mean().detach().cpu()),
        }


def decode_publish_worker(
    ip: str,
    spec: StreamSpec,
    gpu_id: int,
    num_slots: int,
    shm_prefix: str,
    stop_event: threading.Event,
    run_local_ai: bool,
    model: torch.nn.Module | None,
) -> None:
    torch.cuda.set_device(gpu_id)

    publisher: CudaIpcRingPublisher | None = None

    try:
        while not stop_event.is_set():
            feeder = SocketFeeder(ip, spec.port, stop_event)

            try:
                demuxer = nvc.CreateDemuxer(feeder.feed_chunk)

                print(
                    f"[{spec.name}] demuxed: "
                    f"{demuxer.Width()}x{demuxer.Height()} "
                    f"codec={demuxer.GetNvCodecId()} fps={demuxer.FrameRate()}",
                    flush=True,
                )

                decoder = nvc.CreateDecoder(
                    gpuid=gpu_id,
                    codec=demuxer.GetNvCodecId(),
                    usedevicememory=True,
                    outputColorType=nvc.OutputColorType.RGBP,
                    latency=nvc.DisplayDecodeLatencyType.LOW,
                )

                frame_count = 0

                for packet in demuxer:
                    if stop_event.is_set():
                        break

                    # Valid for streams without B-frames; simulator and target camera use low-latency GOP.
                    try:
                        packet.decode_flag = nvc.VideoPacketFlag.ENDOFPICTURE
                    except Exception:
                        pass

                    decoded_frames = decoder.Decode(packet)

                    for frame in decoded_frames:
                        if stop_event.is_set():
                            break

                        # Zero-copy import of decoded GPU frame into Torch.
                        decoded = torch.from_dlpack(frame)
                        chw = decoded_tensor_to_chw_uint8(decoded)

                        if publisher is None:
                            publisher = CudaIpcRingPublisher(
                                stream_name=spec.name,
                                first_frame_chw=chw,
                                num_slots=num_slots,
                                gpu_id=gpu_id,
                                shm_prefix=shm_prefix,
                            )
                        else:
                            publisher.publish(chw)

                        if run_local_ai and model is not None:
                            ai_input = preprocess_for_ai(chw)
                            with torch.inference_mode():
                                result = model(ai_input)
                            if frame_count % 30 == 0:
                                print(f"[{spec.name}] local_ai frame={frame_count} result={result}", flush=True)

                        frame_count += 1

                        if frame_count % 120 == 0:
                            print(f"[{spec.name}] published frames={frame_count}", flush=True)

            except Exception as e:
                if not stop_event.is_set():
                    print(f"[{spec.name}] decode/publish error: {e}", flush=True)
                    print(f"[{spec.name}] reconnecting...", flush=True)

            finally:
                feeder.close()

            time.sleep(1.0)

    finally:
        if publisher is not None:
            publisher.close()
        print(f"[{spec.name}] stopped", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decode TCP video on GPU and publish CUDA IPC ring buffers")
    p.add_argument("--ip", default="127.0.0.1", help="camera/simulator IP")
    p.add_argument("--streams", default="rgb", help="comma-separated streams: rgb,left,right")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--num-slots", type=int, default=4)
    p.add_argument("--shm-prefix", default="cuda_ipc_stream")
    p.add_argument("--local-ai", action="store_true", help="also run DummyAI in this decode process")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    names = parse_stream_names(args.streams)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")

    torch.cuda.set_device(args.gpu_id)

    model: torch.nn.Module | None = None
    if args.local_ai:
        model = DummyAI().cuda().eval()

    stop_event = threading.Event()
    threads: list[threading.Thread] = []

    for name in names:
        t = threading.Thread(
            target=decode_publish_worker,
            args=(
                args.ip,
                STREAMS[name],
                args.gpu_id,
                args.num_slots,
                args.shm_prefix,
                stop_event,
                args.local_ai,
                model,
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)

    print(
        "Decode publisher running. "
        "Start consumers, e.g. consumers\\ipc_gl_viewer_pycuda.py --stream rgb. "
        "Press Ctrl+C to stop.",
        flush=True,
    )

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
        stop_event.set()

    for t in threads:
        t.join(timeout=5.0)

    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

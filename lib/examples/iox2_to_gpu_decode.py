#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iox2_encoded_feeder import Iox2EncodedByteFeeder

from com import Iox2CudaIpcRingPublisher
from utils import decoded_tensor_to_chw_uint8

try:
    from tool.tcp_sim_src import StreamSpec
except Exception:
    @dataclass(frozen=True)
    class StreamSpec:
        name: str
        port: int
        codec: str
        is_mono: bool

import torch
import PyNvVideoCodec as nvc


STREAMS: dict[str, StreamSpec] = {
    "rgb": StreamSpec("rgb", 5000, "hevc", False),
    "left": StreamSpec("left", 5001, "h264", True),
    "right": StreamSpec("right", 5002, "h264", True),
}


def service_name_for(prefix: str, stream_name: str) -> str:
    return f"{prefix.rstrip('/')}/{stream_name}"


def decode_publish_worker(
    encoded_service_name: str,
    spec: StreamSpec,
    gpu_id: int,
    num_slots: int,
    output_service_prefix: str,
    stream_info_period_sec: float,
    poll_sec: float,
    max_return_bytes: int,
    verbose_input: bool,
    stop_event: threading.Event,
) -> None:
    torch.cuda.set_device(gpu_id)
    publisher: Iox2CudaIpcRingPublisher | None = None

    try:
        while not stop_event.is_set():
            feeder: Iox2EncodedByteFeeder | None = None
            try:
                feeder = Iox2EncodedByteFeeder(
                    encoded_service_name,
                    stop_event,
                    poll_sec=poll_sec,
                    max_return_bytes=max_return_bytes,
                    verbose=verbose_input,
                )
                print(
                    f"[{spec.name}] consuming iceoryx2 service '{encoded_service_name}' "
                    f"({spec.codec}) -> GPU decode -> CUDA IPC prefix '{output_service_prefix}'",
                    flush=True,
                )

                demuxer = nvc.CreateDemuxer(feeder.feed_chunk)
                print(
                    f"[{spec.name}] demuxed: {demuxer.Width()}x{demuxer.Height()} "
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

                    try:
                        packet.decode_flag = nvc.VideoPacketFlag.ENDOFPICTURE
                    except Exception:
                        pass

                    decoded_frames = decoder.Decode(packet)
                    for frame in decoded_frames:
                        if stop_event.is_set():
                            break

                        decoded = torch.from_dlpack(frame)
                        chw = decoded_tensor_to_chw_uint8(decoded)

                        if publisher is None:
                            publisher = Iox2CudaIpcRingPublisher(
                                stream_name=spec.name,
                                first_frame_chw=chw,
                                num_slots=num_slots,
                                gpu_id=gpu_id,
                                service_prefix=output_service_prefix,
                                stream_info_period_sec=stream_info_period_sec,
                            )
                        else:
                            publisher.publish(chw)

                        frame_count += 1
                        if frame_count % 120 == 0:
                            print(f"[{spec.name}] cuda ipc pub frames={frame_count}", flush=True)

            except Exception as e:
                if not stop_event.is_set():
                    print(f"[{spec.name}] iox2-decode/iox2-cuda-publish error: {e}", flush=True)
                    print(f"[{spec.name}] restarting demux/decode loop...", flush=True)
            finally:
                if feeder is not None:
                    feeder.close()

            if not stop_event.is_set():
                time.sleep(1.0)
    finally:
        if publisher is not None:
            publisher.close()
        print(f"[{spec.name}] stopped", flush=True)


def parse_stream_names(streams_arg: str) -> list[str]:
    names = [s.strip() for s in streams_arg.split(",") if s.strip()]
    if not names:
        raise RuntimeError("No streams selected")

    for name in names:
        if name not in STREAMS:
            raise RuntimeError(f"Unknown stream '{name}'. Valid: {','.join(STREAMS)}")

    return names


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Subscribe to encoded video over iceoryx2, decode on GPU, and publish CUDA IPC metadata over iceoryx2"
    )
    p.add_argument("--streams", default="rgb", help="comma-separated streams: rgb,left,right")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--num-slots", type=int, default=16, help="more slots reduce overwrite risk for slow consumers")

    p.add_argument(
        "--encoded-iox2-prefix",
        default="sim/video",
        help="input iceoryx2 prefix that carries encoded byte chunks from tcp->iox2 republisher",
    )
    p.add_argument(
        "--iox2-prefix",
        default="CudaIpcVideo",
        help="output iceoryx2 service prefix used by Iox2CudaIpcRingPublisher",
    )
    p.add_argument("--stream-info-period-sec", type=float, default=1.0)
    p.add_argument("--poll-sec", type=float, default=0.001, help="subscriber polling sleep when no encoded samples are available")
    p.add_argument(
        "--max-return-bytes",
        type=int,
        default=256 * 1024,
        help="maximum bytes returned per demuxer callback call",
    )
    p.add_argument("--verbose-input", action="store_true", help="print input iceoryx2 receive statistics")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    names = parse_stream_names(args.streams)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    torch.cuda.set_device(args.gpu_id)

    stop_event = threading.Event()
    threads: list[threading.Thread] = []

    for name in names:
        spec = STREAMS[name]
        encoded_service_name = service_name_for(args.encoded_iox2_prefix, name)
        t = threading.Thread(
            target=decode_publish_worker,
            args=(
                encoded_service_name,
                spec,
                args.gpu_id,
                args.num_slots,
                args.iox2_prefix,
                args.stream_info_period_sec,
                args.poll_sec,
                args.max_return_bytes,
                args.verbose_input,
                stop_event,
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)

    print("iceoryx2 encoded-video GPU decode publisher running. Press Ctrl+C to stop.", flush=True)
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

#!/usr/bin/env python3
"""
pynvvideocodec_ai_client.py

Host-side NVIDIA GPU decode + AI skeleton for PoE TCP streams.

Expected camera streams:
  RGB        H.265 / HEVC  tcp://<camera-ip>:5000
  Left mono  H.264         tcp://<camera-ip>:5001
  Right mono H.264         tcp://<camera-ip>:5002

Install:
  pip install pynvvideocodec torch

Run:
  python3 pynvvideocodec_ai_client.py --ip 192.168.1.200 --streams rgb
  python3 pynvvideocodec_ai_client.py --ip 192.168.1.200 --streams rgb,left,right
"""

import argparse
import socket
import threading
import time
from dataclasses import dataclass

import cv2
import torch
import PyNvVideoCodec as nvc


@dataclass
class StreamSpec:
    name: str
    port: int
    codec_hint: str


STREAMS = {
    "rgb": StreamSpec("rgb", 5000, "hevc"),
    "left": StreamSpec("left", 5001, "h264"),
    "right": StreamSpec("right", 5002, "h264"),
}


class SocketFeeder:
    """
    Callback feeder for nvc.CreateDemuxer(callback).

    PyNvVideoCodec gives us a pre-allocated demuxer buffer.
    We fill it from the TCP socket and return the number of bytes copied.

    Returning 0 means EOF, so we only return 0 when stopping or disconnected.
    """

    def __init__(self, ip: str, port: int, stop_event: threading.Event):
        self.ip = ip
        self.port = port
        self.stop_event = stop_event
        self.sock = None

    def connect(self):
        while not self.stop_event.is_set():
            try:
                print(f"[{self.port}] connecting to {self.ip}:{self.port}")
                self.sock = socket.create_connection((self.ip, self.port), timeout=5)
                self.sock.settimeout(1.0)
                print(f"[{self.port}] connected")
                return
            except Exception as e:
                print(f"[{self.port}] connect failed: {e}")
                time.sleep(1)

        raise RuntimeError("Stopped before connecting")

    def feed_chunk(self, demuxer_buffer):
        if self.stop_event.is_set():
            return 0

        if self.sock is None:
            self.connect()

        capacity = len(demuxer_buffer)

        while not self.stop_event.is_set():
            try:
                data = self.sock.recv(capacity)

                if not data:
                    return 0

                n = len(data)
                demuxer_buffer[:n] = data
                return n

            except socket.timeout:
                continue

            except Exception as e:
                print(f"[{self.port}] socket read error: {e}")
                return 0

        return 0

    def close(self):
        try:
            if self.sock is not None:
                self.sock.close()
        except Exception:
            pass


class DummyAI(torch.nn.Module):
    """
    Replace this with your model.

    Input tensor from PyNvVideoCodec should already be CUDA memory.
    For RGBP output, expected shape is usually:
      [3, H, W]
    """

    def forward(self, x):
        # x: [1, C, H, W], float32, CUDA
        # Replace with real inference:
        return {
            "shape": tuple(x.shape),
            "device": str(x.device),
            "mean": float(x.mean().detach().cpu()),
            "thumbnail": torch.nn.functional.interpolate(x,
                (320,320)).clamp(0,1).mul(255.0).to(dtype=torch.uint8).squeeze(0).permute(1,2,0)
        }


def preprocess_for_ai(frame_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert decoded GPU tensor to model input.

    With outputColorType=RGBP:
      frame_tensor is usually uint8 [3, H, W] on cuda:0.

    Returns:
      float32 [1, 3, H, W] in 0..1 range.
    """

    if frame_tensor.ndim == 3:
        # [C,H,W] or [H,W,C]
        if frame_tensor.shape[0] in (1, 3):
            x = frame_tensor
        else:
            # If interleaved RGB [H,W,C], convert to [C,H,W]
            x = frame_tensor.permute(2, 0, 1)
    elif frame_tensor.ndim == 2:
        # Mono [H,W] -> [1,H,W]
        x = frame_tensor.unsqueeze(0)
    else:
        raise RuntimeError(f"Unexpected decoded tensor shape: {frame_tensor.shape}")

    # If mono but your model expects 3 channels, replicate it.
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)

    x = x.unsqueeze(0)          # [1,C,H,W]
    x = x.float().div_(255.0)   # stays on GPU
    return x


def decode_and_infer_worker(
    ip: str,
    spec: StreamSpec,
    gpu_id: int,
    stop_event: threading.Event,
    model: DummyAI,
):
    while not stop_event.is_set():
        feeder = SocketFeeder(ip, spec.port, stop_event)

        try:
            # Buffer demuxing: useful for live/network/custom data sources.
            demuxer = nvc.CreateDemuxer(feeder.feed_chunk)

            print(
                f"[{spec.name}] demuxed stream: "
                f"{demuxer.Width()}x{demuxer.Height()}, "
                f"codec={demuxer.GetNvCodecId()}, "
                f"fps={demuxer.FrameRate()}"
            )

            decoder = nvc.CreateDecoder(
                gpuid=gpu_id,
                codec=demuxer.GetNvCodecId(),
                usedevicememory=True,

                # Important for AI:
                # RGBP gives planar CHW layout, which is convenient for PyTorch.
                outputColorType=nvc.OutputColorType.RGBP,

                # Important because camera encoder should have B-frames disabled.
                latency=nvc.DisplayDecodeLatencyType.LOW,
            )

            frame_count = 0

            for packet in demuxer:
                if stop_event.is_set():
                    break

                # Low-latency mode is valid for streams without B-frames.
                # In our camera code we set encoder.setNumBFrames(0).
                try:
                    packet.decode_flag = nvc.VideoPacketFlag.ENDOFPICTURE
                except Exception:
                    pass

                decoded_frames = decoder.Decode(packet)

                for frame in decoded_frames:
                    if stop_event.is_set():
                        break

                    # Zero-copy import to PyTorch via DLPack.
                    # Result should be a CUDA tensor.
                    tensor = torch.from_dlpack(frame)

                    ai_input = preprocess_for_ai(tensor)

                    with torch.inference_mode():
                        result = model.forward(ai_input)

                    frame_count += 1

                    tn = result["thumbnail"].cpu().numpy()
                    cv2.imshow(f"{spec.name}-thumbnail",tn)
                    cv2.waitKey(1)

                    if frame_count % 30 == 0:
                        print(f"[{spec.name}] frame={frame_count}") #, result={result}

        except Exception as e:
            if not stop_event.is_set():
                print(f"[{spec.name}] decode/infer error: {e}")
                print(f"[{spec.name}] reconnecting...")

        finally:
            feeder.close()

        time.sleep(1)

    print(f"[{spec.name}] stopped")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", required=True, help="PoE camera IP")
    parser.add_argument(
        "--streams",
        default="rgb",
        help="Comma-separated streams: rgb,left,right",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    selected = [s.strip() for s in args.streams.split(",") if s.strip()]
    for s in selected:
        if s not in STREAMS:
            raise RuntimeError(f"Unknown stream '{s}'. Valid: rgb,left,right")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch.")

    torch.cuda.set_device(args.gpu_id)

    model = DummyAI().cuda().eval()

    stop_event = threading.Event()
    threads = []

    for name in selected:
        t = threading.Thread(
            target=decode_and_infer_worker,
            args=(args.ip, STREAMS[name], args.gpu_id, stop_event, model),
            daemon=True,
        )
        t.start()
        threads.append(t)

    print("Running. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()

    for t in threads:
        t.join(timeout=3)

    print("Done.")


if __name__ == "__main__":
    main()
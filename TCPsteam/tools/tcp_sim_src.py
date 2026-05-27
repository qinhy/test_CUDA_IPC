#!/usr/bin/env python3
"""
tcp_sim_src.py

OpenCV-based simulator for PoE encoded TCP streams.

This script behaves like a tiny camera-side source for fast host/client testing:

  RGB        H.265 / HEVC  tcp://<simulator-ip>:5000
  Left mono  H.264         tcp://<simulator-ip>:5001
  Right mono H.264         tcp://<simulator-ip>:5002

OpenCV is used to generate or read frames. ffmpeg is used only as the low-latency
H.264/H.265 encoder because OpenCV Python wheels do not reliably expose H.264/H.265
encoders across platforms.

Examples:
  # Start only RGB HEVC stream on port 5000
  python3 tcp_sim_src.py --streams rgb

  # Start RGB + stereo mono streams on ports 5000/5001/5002
  python3 tcp_sim_src.py --streams rgb,left,right

  # Feed frames from a local video file instead of synthetic OpenCV frames
  python3 tcp_sim_src.py --streams rgb --video-file sample.mp4

Then run your client against localhost or the simulator machine IP:
  python3 pynvvideocodec_ai_client.py --ip 127.0.0.1 --streams rgb
"""

from __future__ import annotations

import argparse
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class StreamSpec:
    name: str
    port: int
    codec: str          # "hevc" or "h264"
    is_mono: bool


STREAMS = {
    "rgb": StreamSpec("rgb", 5000, "hevc", False),
    "left": StreamSpec("left", 5001, "h264", True),
    "right": StreamSpec("right", 5002, "h264", True),
}


class FrameSource:
    """OpenCV frame source that can generate synthetic frames or read a video file."""

    def __init__(
        self,
        spec: StreamSpec,
        width: int,
        height: int,
        fps: float,
        video_file: Optional[str] = None,
        loop_video: bool = True,
    ):
        self.spec = spec
        self.width = width
        self.height = height
        self.fps = fps
        self.video_file = video_file
        self.loop_video = loop_video
        self.frame_idx = 0
        self.cap: Optional[cv2.VideoCapture] = None

        if video_file:
            self.cap = cv2.VideoCapture(video_file)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video file: {video_file}")

    def read(self) -> np.ndarray:
        """
        Return one frame for ffmpeg stdin.

        RGB stream returns BGR uint8 shape [H, W, 3].
        Mono streams return GRAY uint8 shape [H, W].
        """
        if self.cap is not None:
            frame = self._read_video_frame()
        else:
            frame = self._make_synthetic_frame()

        self.frame_idx += 1
        return frame

    def _read_video_frame(self) -> np.ndarray:
        assert self.cap is not None

        ok, frame = self.cap.read()
        if not ok:
            if not self.loop_video:
                raise EOFError("Video file ended")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                raise EOFError("Video file ended and could not loop")

        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        if self.spec.is_mono:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = self._apply_stereo_offset(frame)
            self._draw_gray_overlay(frame)
            return frame

        self._draw_bgr_overlay(frame)
        return frame

    def _make_synthetic_frame(self) -> np.ndarray:
        t = self.frame_idx / max(self.fps, 1.0)

        if self.spec.is_mono:
            yy, xx = np.indices((self.height, self.width), dtype=np.int32)
            phase = int(self.frame_idx * 4)
            img = ((xx * 2 + yy + phase) % 256).astype(np.uint8)

            # Simulate a stereo pair by shifting the moving objects differently.
            img = self._apply_stereo_offset(img)

            cx = int((self.width * 0.5) + np.sin(t * 1.4) * self.width * 0.25)
            cy = int((self.height * 0.5) + np.cos(t * 1.1) * self.height * 0.25)
            cv2.circle(img, (cx, cy), max(12, self.height // 14), 235, -1)
            cv2.rectangle(
                img,
                (max(0, cx - 90), max(0, cy - 30)),
                (min(self.width - 1, cx + 90), min(self.height - 1, cy + 30)),
                80,
                3,
            )
            self._draw_gray_overlay(img)
            return img

        x = np.linspace(0, 255, self.width, dtype=np.uint8)
        y = np.linspace(0, 255, self.height, dtype=np.uint8)
        xv = np.tile(x, (self.height, 1))
        yv = np.tile(y[:, None], (1, self.width))

        # OpenCV uses BGR order.
        frame = np.empty((self.height, self.width, 3), dtype=np.uint8)
        frame[..., 0] = (xv.astype(np.uint16) + self.frame_idx * 3) % 256
        frame[..., 1] = (yv.astype(np.uint16) + self.frame_idx * 2) % 256
        frame[..., 2] = ((xv.astype(np.uint16) // 2 + yv.astype(np.uint16) // 2 + self.frame_idx * 5) % 256)

        cx = int((self.width * 0.5) + np.sin(t * 1.2) * self.width * 0.30)
        cy = int((self.height * 0.5) + np.cos(t * 0.9) * self.height * 0.28)
        radius = max(20, min(self.width, self.height) // 10)

        cv2.circle(frame, (cx, cy), radius, (0, 0, 255), -1)
        cv2.rectangle(
            frame,
            (max(0, cx - radius * 2), max(0, cy - radius)),
            (min(self.width - 1, cx + radius * 2), min(self.height - 1, cy + radius)),
            (255, 0, 0),
            4,
        )
        cv2.line(frame, (0, cy), (self.width - 1, self.height - cy - 1), (255, 255, 255), 2)
        self._draw_bgr_overlay(frame)
        return frame

    def _apply_stereo_offset(self, gray: np.ndarray) -> np.ndarray:
        if self.spec.name == "left":
            dx = -12
        elif self.spec.name == "right":
            dx = 12
        else:
            dx = 0

        if dx == 0:
            return gray

        matrix = np.float32([[1, 0, dx], [0, 1, 0]])
        return cv2.warpAffine(
            gray,
            matrix,
            (self.width, self.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    def _draw_bgr_overlay(self, frame: np.ndarray) -> None:
        msg = f"SIM {self.spec.name.upper()}  frame={self.frame_idx:06d}"
        cv2.putText(frame, msg, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, msg, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_gray_overlay(self, frame: np.ndarray) -> None:
        msg = f"SIM {self.spec.name.upper()} frame={self.frame_idx:06d}"
        cv2.putText(frame, msg, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 0, 5, cv2.LINE_AA)
        cv2.putText(frame, msg, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2, cv2.LINE_AA)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


def build_ffmpeg_command(
    ffmpeg_bin: str,
    spec: StreamSpec,
    width: int,
    height: int,
    fps: float,
    bitrate: str,
    gop: int,
    preset: str,
    loglevel: str,
) -> list[str]:
    input_pix_fmt = "gray" if spec.is_mono else "bgr24"

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel", loglevel,
        "-f", "rawvideo",
        "-pix_fmt", input_pix_fmt,
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-an",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
    ]

    if spec.codec == "h264":
        cmd += [
            "-c:v", "libx264",
            "-preset", preset,
            "-tune", "zerolatency",
            "-bf", "0",
            "-g", str(gop),
            "-keyint_min", str(gop),
            "-b:v", bitrate,
            # repeat-headers makes reconnecting easier because SPS/PPS appears on keyframes.
            "-x264-params", f"scenecut=0:repeat-headers=1:keyint={gop}:min-keyint={gop}:bframes=0",
            "-pix_fmt", "yuv420p",
            "-f", "h264",
            "pipe:1",
        ]
    elif spec.codec == "hevc":
        cmd += [
            "-c:v", "libx265",
            "-preset", preset,
            "-tune", "zerolatency",
            "-bf", "0",
            "-g", str(gop),
            "-keyint_min", str(gop),
            "-b:v", bitrate,
            # repeat-headers makes reconnecting easier because VPS/SPS/PPS appears on keyframes.
            "-x265-params", f"log-level=error:scenecut=0:repeat-headers=1:keyint={gop}:min-keyint={gop}:bframes=0",
            "-pix_fmt", "yuv420p",
            "-f", "hevc",
            "pipe:1",
        ]
    else:
        raise RuntimeError(f"Unsupported codec: {spec.codec}")

    return cmd


def drain_stderr(proc: subprocess.Popen, stream_name: str, stop_event: threading.Event) -> None:
    if proc.stderr is None:
        return
    while not stop_event.is_set():
        line = proc.stderr.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace").rstrip()
        if text:
            print(f"[{stream_name}] ffmpeg: {text}")


def feed_frames_to_encoder(
    proc: subprocess.Popen,
    source: FrameSource,
    fps: float,
    stop_event: threading.Event,
) -> None:
    next_frame_time = time.perf_counter()
    frame_interval = 1.0 / max(fps, 1.0)

    try:
        while not stop_event.is_set():
            try:
                frame = source.read()
            except EOFError:
                break

            if proc.stdin is None:
                break

            try:
                proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                break

            next_frame_time += frame_interval
            sleep_for = next_frame_time - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # If encoding falls behind, do not accumulate a huge delay.
                next_frame_time = time.perf_counter()

    finally:
        source.close()
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass


def handle_client(conn: socket.socket, addr, spec: StreamSpec, args: argparse.Namespace) -> None:
    width = args.mono_width if spec.is_mono else args.rgb_width
    height = args.mono_height if spec.is_mono else args.rgb_height
    bitrate = args.mono_bitrate if spec.is_mono else args.rgb_bitrate
    gop = max(1, int(round(args.fps * args.keyframe_seconds)))

    print(f"[{spec.name}] client connected from {addr[0]}:{addr[1]}")

    cmd = build_ffmpeg_command(
        ffmpeg_bin=args.ffmpeg_bin,
        spec=spec,
        width=width,
        height=height,
        fps=args.fps,
        bitrate=bitrate,
        gop=gop,
        preset=args.encoder_preset,
        loglevel=args.ffmpeg_loglevel,
    )

    proc_stop = threading.Event()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    source = FrameSource(
        spec=spec,
        width=width,
        height=height,
        fps=args.fps,
        video_file=args.video_file,
        loop_video=not args.no_loop_video,
    )

    feeder = threading.Thread(
        target=feed_frames_to_encoder,
        args=(proc, source, args.fps, proc_stop),
        daemon=True,
    )
    stderr_reader = threading.Thread(
        target=drain_stderr,
        args=(proc, spec.name, proc_stop),
        daemon=True,
    )
    feeder.start()
    stderr_reader.start()

    byte_count = 0
    start_time = time.perf_counter()

    try:
        if proc.stdout is None:
            raise RuntimeError("ffmpeg stdout was not created")

        while not proc_stop.is_set():
            chunk = proc.stdout.read(64 * 1024)
            if not chunk:
                break

            try:
                conn.sendall(chunk)
            except (BrokenPipeError, ConnectionResetError, OSError):
                break

            byte_count += len(chunk)
            now = time.perf_counter()
            if now - start_time >= 5.0:
                mbps = (byte_count * 8.0) / (now - start_time) / 1_000_000.0
                print(f"[{spec.name}] streaming {spec.codec} {width}x{height}@{args.fps:g}, {mbps:.2f} Mbps")
                byte_count = 0
                start_time = now

    finally:
        proc_stop.set()
        try:
            conn.close()
        except Exception:
            pass

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()

        feeder.join(timeout=1.0)
        stderr_reader.join(timeout=1.0)
        print(f"[{spec.name}] client disconnected")


def serve_stream(spec: StreamSpec, args: argparse.Namespace, stop_event: threading.Event) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, spec.port))
        server.listen(1)
        server.settimeout(1.0)

        print(f"[{spec.name}] listening on tcp://{args.host}:{spec.port} as {spec.codec}")

        while not stop_event.is_set():
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass

            # One active client per stream keeps timing and reconnection behavior simple.
            handle_client(conn, addr, spec, args)

        print(f"[{spec.name}] server stopped")


def parse_streams(streams_arg: str) -> list[StreamSpec]:
    names = [s.strip() for s in streams_arg.split(",") if s.strip()]
    if not names:
        raise RuntimeError("No streams selected")

    specs = []
    for name in names:
        if name not in STREAMS:
            raise RuntimeError(f"Unknown stream '{name}'. Valid streams: {','.join(STREAMS)}")
        specs.append(STREAMS[name])
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenCV TCP simulator for encoded streams")
    parser.add_argument("--host", default="0.0.0.0", help="TCP bind address")
    parser.add_argument("--streams", default="rgb", help="Comma-separated streams: rgb,left,right")
    parser.add_argument("--fps", type=float, default=30.0)

    parser.add_argument("--rgb-width", type=int, default=1280)
    parser.add_argument("--rgb-height", type=int, default=720)
    parser.add_argument("--mono-width", type=int, default=640)
    parser.add_argument("--mono-height", type=int, default=400)

    parser.add_argument("--rgb-bitrate", default="6M")
    parser.add_argument("--mono-bitrate", default="2M")
    parser.add_argument("--keyframe-seconds", type=float, default=1.0)

    parser.add_argument("--video-file", default=None, help="Optional OpenCV-readable video file to stream")
    parser.add_argument("--no-loop-video", action="store_true", help="Do not loop --video-file at EOF")

    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffmpeg-loglevel", default="warning", choices=["quiet", "error", "warning", "info", "verbose"])
    parser.add_argument("--encoder-preset", default="ultrafast")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = parse_streams(args.streams)

    if shutil.which(args.ffmpeg_bin) is None:
        print(
            f"ERROR: '{args.ffmpeg_bin}' was not found. Install ffmpeg or pass --ffmpeg-bin /path/to/ffmpeg.",
            file=sys.stderr,
        )
        return 2

    stop_event = threading.Event()

    def request_stop(signum, _frame):
        print(f"\nReceived signal {signum}; stopping...")
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    threads = []
    for spec in specs:
        thread = threading.Thread(target=serve_stream, args=(spec, args, stop_event), daemon=True)
        thread.start()
        threads.append(thread)

    print("Simulator running. Connect your decoder/client to this machine's IP.")

    while not stop_event.is_set():
        time.sleep(0.2)

    for thread in threads:
        thread.join(timeout=2.0)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

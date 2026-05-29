#!/usr/bin/env python3
"""OAK-D PoE-like encoded stream simulator.

This module intentionally mimics the most useful DepthAI behavior for this
experiment: the host receives complete encoded packets/messages, not random TCP
chunks. The actual image generation is OpenCV, and the actual H264/H265 encoder
is FFmpeg/libx264/libx265.
"""
from __future__ import annotations

import argparse
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class StreamSpec:
    name: str
    stream_id: int
    codec: str          # "hevc" or "h264"
    is_mono: bool
    width: int
    height: int
    bitrate: str


DEFAULT_STREAMS = {
    "rgb": StreamSpec("rgb", 0, "hevc", False, 1280*2, 720*2, "6M"),
    "left": StreamSpec("left", 1, "h264", True, 640, 400, "2M"),
    "right": StreamSpec("right", 2, "h264", True, 640, 400, "2M"),
}


@dataclass
class EncodedPacket:
    """DepthAI-like encoded output message."""
    stream_name: str
    stream_id: int
    sequence_number: int
    timestamp_ns: int
    codec: str
    width: int
    height: int
    fps: float
    payload: bytes       # complete Annex-B access unit including start codes
    is_keyframe: bool
    has_headers: bool

    def getData(self) -> bytes:
        """DepthAI-like API convenience."""
        return self.payload

    def getTimestampNs(self) -> int:
        return self.timestamp_ns

    def getSequenceNum(self) -> int:
        return self.sequence_number


class FrameGenerator:
    def __init__(self, spec: StreamSpec, fps: float, video_file: Optional[str] = None, loop_video: bool = True):
        self.spec = spec
        self.fps = fps
        self.video_file = video_file
        self.loop_video = loop_video
        self.frame_idx = 0
        self.cap: Optional[cv2.VideoCapture] = None
        if video_file:
            self.cap = cv2.VideoCapture(video_file)
            if not self.cap.isOpened():
                raise RuntimeError(f"could not open video file: {video_file}")

    def read(self) -> np.ndarray:
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
                raise EOFError("video file ended")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                raise EOFError("video file ended and could not loop")
        frame = cv2.resize(frame, (self.spec.width, self.spec.height), interpolation=cv2.INTER_LINEAR)
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
            yy, xx = np.indices((self.spec.height, self.spec.width), dtype=np.int32)
            phase = int(self.frame_idx * 4)
            img = ((xx * 2 + yy + phase) % 256).astype(np.uint8)
            img = self._apply_stereo_offset(img)
            cx = int((self.spec.width * 0.5) + np.sin(t * 1.4) * self.spec.width * 0.25)
            cy = int((self.spec.height * 0.5) + np.cos(t * 1.1) * self.spec.height * 0.25)
            cv2.circle(img, (cx, cy), max(12, self.spec.height // 14), 235, -1)
            cv2.rectangle(img, (max(0, cx - 90), max(0, cy - 30)),
                          (min(self.spec.width - 1, cx + 90), min(self.spec.height - 1, cy + 30)), 80, 3)
            self._draw_gray_overlay(img)
            return img

        x = np.linspace(0, 255, self.spec.width, dtype=np.uint8)
        y = np.linspace(0, 255, self.spec.height, dtype=np.uint8)
        xv = np.tile(x, (self.spec.height, 1))
        yv = np.tile(y[:, None], (1, self.spec.width))
        # frame = np.random.randint(0,255,(self.spec.height, self.spec.width, 3), dtype=np.uint8)#
        frame = np.empty((self.spec.height, self.spec.width, 3), dtype=np.uint8)
        frame[..., 0] = (xv.astype(np.uint16) + self.frame_idx * 3) % 256
        frame[..., 1] = (yv.astype(np.uint16) + self.frame_idx * 2) % 256
        frame[..., 2] = ((xv.astype(np.uint16) // 2 + yv.astype(np.uint16) // 2 + self.frame_idx * 5) % 256)
        cx = int((self.spec.width * 0.5) + np.sin(t * 1.2) * self.spec.width * 0.30)
        cy = int((self.spec.height * 0.5) + np.cos(t * 0.9) * self.spec.height * 0.28)
        radius = max(20, min(self.spec.width, self.spec.height) // 10)
        cv2.circle(frame, (cx, cy), radius, (0, 0, 255), -1)
        cv2.rectangle(frame, (max(0, cx - radius * 2), max(0, cy - radius)),
                      (min(self.spec.width - 1, cx + radius * 2), min(self.spec.height - 1, cy + radius)), (255, 0, 0), 4)
        cv2.line(frame, (0, cy), (self.spec.width - 1, self.spec.height - cy - 1), (255, 255, 255), 2)
        self._draw_bgr_overlay(frame)
        return frame

    def _apply_stereo_offset(self, gray: np.ndarray) -> np.ndarray:
        dx = -12 if self.spec.name == "left" else 12 if self.spec.name == "right" else 0
        if dx == 0:
            return gray
        matrix = np.float32([[1, 0, dx], [0, 1, 0]])
        return cv2.warpAffine(gray, matrix, (self.spec.width, self.spec.height), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)

    def _draw_bgr_overlay(self, frame: np.ndarray) -> None:
        msg = f"OAK-D-POE SIM {self.spec.name.upper()} frame={self.frame_idx:06d}"
        cv2.putText(frame, msg, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, msg, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_gray_overlay(self, frame: np.ndarray) -> None:
        msg = f"OAK-D-POE SIM {self.spec.name.upper()} frame={self.frame_idx:06d}"
        cv2.putText(frame, msg, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 0, 5, cv2.LINE_AA)
        cv2.putText(frame, msg, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2, cv2.LINE_AA)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


def build_ffmpeg_command(ffmpeg_bin: str, spec: StreamSpec, fps: float, gop: int, preset: str, loglevel: str) -> list[str]:
    input_pix_fmt = "gray" if spec.is_mono else "bgr24"
    cmd = [
        ffmpeg_bin, "-hide_banner", "-loglevel", loglevel,
        "-f", "rawvideo", "-pix_fmt", input_pix_fmt,
        "-s", f"{spec.width}x{spec.height}", "-r", str(fps), "-i", "pipe:0",
        "-an", "-fflags", "nobuffer", "-flags", "low_delay",
    ]
    if spec.codec == "h264":
        cmd += [
            "-c:v", "libx264", "-preset", preset, "-tune", "zerolatency",
            "-bf", "0", "-g", str(gop), "-keyint_min", str(gop), "-b:v", spec.bitrate,
            "-x264-params", f"scenecut=0:repeat-headers=1:aud=1:keyint={gop}:min-keyint={gop}:bframes=0",
            "-pix_fmt", "yuv420p", "-f", "h264", "pipe:1",
        ]
    elif spec.codec == "hevc":
        cmd += [
            "-c:v", "libx265", "-preset", preset, "-tune", "zerolatency",
            "-bf", "0", "-g", str(gop), "-keyint_min", str(gop), "-b:v", spec.bitrate,
            "-x265-params", f"log-level=error:scenecut=0:repeat-headers=1:aud=1:keyint={gop}:min-keyint={gop}:bframes=0",
            "-pix_fmt", "yuv420p", "-f", "hevc", "pipe:1",
        ]
    else:
        raise ValueError(f"unsupported codec: {spec.codec}")
    return cmd


def _find_start_codes(buf: bytes) -> list[int]:
    starts = []
    i = 0
    n = len(buf)
    while i < n - 3:
        if buf[i:i+3] == b"\x00\x00\x01":
            starts.append(i); i += 3
        elif i < n - 4 and buf[i:i+4] == b"\x00\x00\x00\x01":
            starts.append(i); i += 4
        else:
            i += 1
    return starts


class AnnexBNalSplitter:
    def __init__(self):
        self.buf = bytearray()

    def push(self, data: bytes):
        self.buf.extend(data)
        out = []
        while True:
            starts = _find_start_codes(bytes(self.buf))
            if len(starts) < 2:
                # Keep only enough leading junk to find a future start code.
                if len(starts) == 0 and len(self.buf) > 4:
                    del self.buf[:-4]
                break
            first, second = starts[0], starts[1]
            if first > 0:
                del self.buf[:first]
                second -= first
            nal = bytes(self.buf[:second])
            del self.buf[:second]
            if nal:
                out.append(nal)
        return out


def _payload_offset(nal: bytes) -> int:
    if nal.startswith(b"\x00\x00\x00\x01"):
        return 4
    if nal.startswith(b"\x00\x00\x01"):
        return 3
    return 0


def nal_type(codec: str, nal: bytes) -> int:
    off = _payload_offset(nal)
    if len(nal) <= off:
        return -1
    if codec == "h264":
        return nal[off] & 0x1F
    return (nal[off] >> 1) & 0x3F


class AccessUnitGrouper:
    def __init__(self, codec: str):
        self.codec = codec
        self.current: list[bytes] = []

    def push(self, nal: bytes) -> Optional[tuple[bytes, bool, bool]]:
        nt = nal_type(self.codec, nal)
        aud_type = 9 if self.codec == "h264" else 35
        if nt == aud_type and self.current:
            au = self._finish()
            self.current.append(nal)
            return au
        self.current.append(nal)
        return None

    def _finish(self) -> tuple[bytes, bool, bool]:
        payload = b"".join(self.current)
        types = {nal_type(self.codec, n) for n in self.current}
        if self.codec == "h264":
            is_keyframe = 5 in types
            has_headers = 7 in types and 8 in types
        else:
            is_keyframe = bool(types.intersection({19, 20, 21}))
            has_headers = 32 in types and 33 in types and 34 in types
        self.current = []
        return payload, is_keyframe, has_headers


def _drain_stderr(proc: subprocess.Popen, name: str, stop: threading.Event) -> None:
    if proc.stderr is None:
        return
    while not stop.is_set():
        line = proc.stderr.readline()
        if not line:
            break
        text = line.decode("utf-8", "replace").rstrip()
        if text:
            print(f"[{name}] ffmpeg: {text}")


class SimulatedOutputQueue:
    def __init__(self, name: str, q: "queue.Queue[EncodedPacket]", blocking: bool = True):
        self.name = name
        self._q = q
        self.blocking = blocking

    def get(self, timeout: Optional[float] = None) -> EncodedPacket:
        return self._q.get(block=self.blocking, timeout=timeout)

    def tryGet(self) -> Optional[EncodedPacket]:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None


class OakdPoeLikeDevice:
    """Small DepthAI-like device object exposing encoded output queues."""
    def __init__(self, specs: list[StreamSpec], fps: float = 30.0, keyframe_seconds: float = 1.0,
                 ffmpeg_bin: str = "ffmpeg", encoder_preset: str = "ultrafast",
                 ffmpeg_loglevel: str = "warning", queue_size: int = 30,
                 video_file: Optional[str] = None, loop_video: bool = True):
        if shutil.which(ffmpeg_bin) is None:
            raise RuntimeError(f"'{ffmpeg_bin}' not found")
        self.specs = specs
        self.fps = fps
        self.keyframe_seconds = keyframe_seconds
        self.ffmpeg_bin = ffmpeg_bin
        self.encoder_preset = encoder_preset
        self.ffmpeg_loglevel = ffmpeg_loglevel
        self.queue_size = queue_size
        self.video_file = video_file
        self.loop_video = loop_video
        self.stop = threading.Event()
        self.queues: dict[str, queue.Queue[EncodedPacket]] = {s.name: queue.Queue(maxsize=queue_size) for s in specs}
        self.threads: list[threading.Thread] = []

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def start(self) -> None:
        for spec in self.specs:
            t = threading.Thread(target=self._run_stream, args=(spec,), daemon=True)
            t.start()
            self.threads.append(t)

    def getOutputQueue(self, name: str, maxSize: Optional[int] = None, blocking: bool = True) -> SimulatedOutputQueue:
        # maxSize is accepted for DepthAI API similarity; queue size is configured at construction time.
        return SimulatedOutputQueue(name, self.queues[name], blocking=blocking)

    def close(self) -> None:
        self.stop.set()
        for t in self.threads:
            t.join(timeout=2.0)

    def _run_stream(self, spec: StreamSpec) -> None:
        gop = max(1, int(round(self.fps * self.keyframe_seconds)))
        cmd = build_ffmpeg_command(self.ffmpeg_bin, spec, self.fps, gop, self.encoder_preset, self.ffmpeg_loglevel)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        stderr_t = threading.Thread(target=_drain_stderr, args=(proc, spec.name, self.stop), daemon=True)
        stderr_t.start()
        gen = FrameGenerator(spec, self.fps, self.video_file, self.loop_video)
        splitter = AnnexBNalSplitter()
        grouper = AccessUnitGrouper(spec.codec)
        seq = 0
        next_frame = time.perf_counter()
        frame_interval = 1.0 / max(self.fps, 1.0)

        def feed_frames():
            nonlocal next_frame
            try:
                while not self.stop.is_set():
                    frame = gen.read()
                    if proc.stdin is None:
                        break
                    proc.stdin.write(frame.tobytes())
                    next_frame += frame_interval
                    delay = next_frame - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_frame = time.perf_counter()
            except (BrokenPipeError, OSError, EOFError):
                pass
            finally:
                gen.close()
                try:
                    if proc.stdin:
                        proc.stdin.close()
                except Exception:
                    pass

        feeder = threading.Thread(target=feed_frames, daemon=True)
        feeder.start()
        try:
            assert proc.stdout is not None
            while not self.stop.is_set():
                chunk = proc.stdout.read(32 * 1024)
                if not chunk:
                    break
                for nal in splitter.push(chunk):
                    au = grouper.push(nal)
                    if au is None:
                        continue
                    payload, keyframe, headers = au
                    pkt = EncodedPacket(spec.name, spec.stream_id, seq, time.time_ns(), spec.codec,
                                        spec.width, spec.height, self.fps, payload, keyframe, headers)
                    seq += 1
                    q = self.queues[spec.name]
                    # Simulate DepthAI host queue with drop-oldest behavior when non-blocking/latency-oriented.
                    if q.full():
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            pass
                    q.put(pkt)
        finally:
            self.stop.set()
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
            feeder.join(timeout=1.0)
            stderr_t.join(timeout=1.0)


def parse_stream_specs(names: str) -> list[StreamSpec]:
    out = []
    for name in [x.strip() for x in names.split(",") if x.strip()]:
        if name not in DEFAULT_STREAMS:
            raise ValueError(f"unknown stream '{name}', valid: {','.join(DEFAULT_STREAMS)}")
        out.append(DEFAULT_STREAMS[name])
    if not out:
        raise ValueError("no streams selected")
    return out


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--streams", default="rgb", help="comma-separated: rgb,left,right")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--keyframe-seconds", type=float, default=0.5)
    p.add_argument("--queue-size", type=int, default=30)
    p.add_argument("--video-file", default=None)
    p.add_argument("--no-loop-video", action="store_true")
    p.add_argument("--ffmpeg-bin", default="ffmpeg")
    p.add_argument("--ffmpeg-loglevel", default="warning", choices=["quiet", "error", "warning", "info", "verbose"])
    p.add_argument("--encoder-preset", default="ultrafast")

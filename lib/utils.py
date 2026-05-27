from __future__ import annotations

import os
import socket
import struct
import threading
import time
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import torch

MAGIC = b"CIPCSTRM"
VERSION = 1

# Keep this small and fixed-size so C/C++ ports are easy later.
# magic:8s
# version:uint32
# stream_name:32s
# layout:32s
# width:uint32
# height:uint32
# channels:uint32
# num_slots:uint32
# itemsize:uint64
# frame_bytes:uint64
# handle_len:uint64
# producer_pid:uint64
HEADER_FMT = "<8sI32s32sIIIIQQQQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

HANDLE_OFFSET = 512
CTRL_OFFSET = 1024

LATEST_OFFSET = CTRL_OFFSET          # int64 latest complete seq, starts at -1
STOP_OFFSET = CTRL_OFFSET + 8        # int64 0/1
GEN_OFFSET = CTRL_OFFSET + 16        # int64[num_slots], slot generation counters

# Enough for metadata + 64-byte CUDA IPC handle + many slots.
DEFAULT_SHM_SIZE = 4096

LAYOUT_RGBP_CHW = "RGBP_CHW"
LAYOUT_GRAY_CHW = "GRAY_CHW"
LAYOUT_RGBA_CHW = "RGBA_CHW"

DTYPE_UINT8 = "uint8"


@dataclass(frozen=True)
class StreamHeader:
    stream_name: str
    layout: str
    width: int
    height: int
    channels: int
    num_slots: int
    itemsize: int
    frame_bytes: int
    handle_len: int
    producer_pid: int

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self.num_slots, self.channels, self.height, self.width)


def shm_name_for(stream_name: str, prefix: str = "cuda_ipc_stream") -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in stream_name)
    return f"{prefix}_{safe}_v1"


def _enc_fixed(text: str, size: int) -> bytes:
    raw = text.encode("utf-8")
    if len(raw) >= size:
        raw = raw[: size - 1]
    return raw + b"\x00" * (size - len(raw))


def _dec_fixed(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")


def pack_header(h: StreamHeader) -> bytes:
    return struct.pack(
        HEADER_FMT,
        MAGIC,
        VERSION,
        _enc_fixed(h.stream_name, 32),
        _enc_fixed(h.layout, 32),
        int(h.width),
        int(h.height),
        int(h.channels),
        int(h.num_slots),
        int(h.itemsize),
        int(h.frame_bytes),
        int(h.handle_len),
        int(h.producer_pid),
    )


def unpack_header(data: bytes | memoryview) -> StreamHeader:
    tup = struct.unpack(HEADER_FMT, bytes(data[:HEADER_SIZE]))
    magic, version, stream_name, layout, width, height, channels, num_slots, itemsize, frame_bytes, handle_len, producer_pid = tup

    if magic != MAGIC:
        raise RuntimeError(f"bad shared-memory magic: {magic!r}")
    if version != VERSION:
        raise RuntimeError(f"unsupported shared-memory version: {version}")

    return StreamHeader(
        stream_name=_dec_fixed(stream_name),
        layout=_dec_fixed(layout),
        width=int(width),
        height=int(height),
        channels=int(channels),
        num_slots=int(num_slots),
        itemsize=int(itemsize),
        frame_bytes=int(frame_bytes),
        handle_len=int(handle_len),
        producer_pid=int(producer_pid),
    )


def layout_for_channels(channels: int) -> str:
    if channels == 1:
        return LAYOUT_GRAY_CHW
    if channels == 3:
        return LAYOUT_RGBP_CHW
    if channels == 4:
        return LAYOUT_RGBA_CHW
    raise RuntimeError(f"unsupported channel count for IPC layout: {channels}")

def service_base_name(prefix: str, stream_name: str) -> str:
    prefix = prefix.strip("/").replace("\\", "/")
    stream = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in stream_name)
    return f"{prefix}/{stream}"


def stream_info_service_name(prefix: str, stream_name: str) -> str:
    return f"{service_base_name(prefix, stream_name)}/StreamInfo"


def frame_ready_service_name(prefix: str, stream_name: str) -> str:
    return f"{service_base_name(prefix, stream_name)}/FrameReady"


def now_ns() -> int:
    return time.time_ns()


def decoded_tensor_to_chw_uint8(t: torch.Tensor) -> torch.Tensor:
    """
    Normalize a decoded PyNvVideoCodec tensor to CUDA uint8 CHW.

    Accepts:
      [C,H,W], [H,W,C], or [H,W]
    Returns:
      contiguous uint8 [C,H,W] on CUDA, with C in {1,3,4}.
    """
    if not t.is_cuda:
        raise RuntimeError(f"expected CUDA tensor, got device={t.device}")

    if t.dtype != torch.uint8:
        t = t.to(dtype=torch.uint8)

    if t.ndim == 2:
        x = t.unsqueeze(0)

    elif t.ndim == 3:
        # Planar RGBP/CHW from PyNvVideoCodec is the intended fast path.
        if t.shape[0] in (1, 3, 4):
            x = t
        elif t.shape[-1] in (1, 3, 4):
            x = t.permute(2, 0, 1)
        else:
            raise RuntimeError(f"cannot infer channel dimension from shape={tuple(t.shape)}")
    else:
        raise RuntimeError(f"unexpected decoded tensor shape={tuple(t.shape)}")

    if x.shape[0] not in (1, 3, 4):
        raise RuntimeError(f"expected 1, 3, or 4 channels, got shape={tuple(x.shape)}")

    return x.contiguous()


def torch_tensor_to_cupy_view(t: torch.Tensor):
    """
    Return a CuPy view of a CUDA torch tensor using DLPack.
    The returned CuPy array aliases the torch tensor memory.
    """
    import cupy as cp

    if not t.is_cuda:
        raise RuntimeError("torch tensor must be CUDA")

    # Newer CuPy supports Python DLPack protocol directly.
    try:
        return cp.from_dlpack(t)
    except Exception:
        # Compatibility fallback.
        return cp.fromDlpack(torch.utils.dlpack.to_dlpack(t))


def now_ns() -> int:
    return time.time_ns()


class SocketFeeder:
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


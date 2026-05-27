from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Optional


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


def write_header(buf: memoryview, h: StreamHeader) -> None:
    buf[:HEADER_SIZE] = pack_header(h)


def read_header(buf: memoryview) -> StreamHeader:
    return unpack_header(buf)


def write_i64(buf: memoryview, offset: int, value: int) -> None:
    struct.pack_into("<q", buf, offset, int(value))


def read_i64(buf: memoryview, offset: int) -> int:
    return struct.unpack_from("<q", buf, offset)[0]


def gen_offset(slot: int) -> int:
    return GEN_OFFSET + int(slot) * 8


def initialize_control(buf: memoryview, num_slots: int) -> None:
    write_i64(buf, LATEST_OFFSET, -1)
    write_i64(buf, STOP_OFFSET, 0)
    for slot in range(num_slots):
        write_i64(buf, gen_offset(slot), 0)


def mark_stopped(buf: memoryview) -> None:
    write_i64(buf, STOP_OFFSET, 1)


def create_clean_shm(name: str, size: int = DEFAULT_SHM_SIZE) -> SharedMemory:
    try:
        old = SharedMemory(name=name, create=False)
        old.close()
        old.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        # Probably still in use. Let create=True fail with a useful error.
        pass

    return SharedMemory(name=name, create=True, size=size)


def open_shm_wait(name: str, timeout_sec: float = 30.0, poll_sec: float = 0.2) -> SharedMemory:
    deadline = time.time() + timeout_sec
    last_err: Optional[BaseException] = None

    while time.time() < deadline:
        try:
            return SharedMemory(name=name, create=False)
        except FileNotFoundError as e:
            last_err = e
            time.sleep(poll_sec)

    raise TimeoutError(f"timed out waiting for shared memory '{name}': {last_err}")


def wait_for_valid_header(
    name: str,
    timeout_sec: float = 30.0,
    poll_sec: float = 0.2,
) -> tuple[SharedMemory, StreamHeader]:
    deadline = time.time() + timeout_sec
    last_err: Optional[BaseException] = None

    while time.time() < deadline:
        try:
            shm = SharedMemory(name=name, create=False)
            try:
                h = read_header(shm.buf)
                if h.handle_len <= 0:
                    raise RuntimeError("header exists but handle_len is zero")
                return shm, h
            except Exception:
                shm.close()
                raise
        except Exception as e:
            last_err = e
            time.sleep(poll_sec)

    raise TimeoutError(f"timed out waiting for valid header in '{name}': {last_err}")


def get_handle_bytes(buf: memoryview, handle_len: int) -> bytes:
    if handle_len <= 0:
        raise RuntimeError(f"invalid CUDA IPC handle length: {handle_len}")
    if HANDLE_OFFSET + handle_len > CTRL_OFFSET:
        raise RuntimeError("CUDA IPC handle overlaps control region")
    return bytes(buf[HANDLE_OFFSET : HANDLE_OFFSET + handle_len])


def write_handle_bytes(buf: memoryview, handle: bytes | bytearray) -> None:
    handle = bytes(handle)
    if HANDLE_OFFSET + len(handle) > CTRL_OFFSET:
        raise RuntimeError("CUDA IPC handle does not fit in shared-memory layout")
    buf[HANDLE_OFFSET : HANDLE_OFFSET + len(handle)] = handle


def expected_generation(seq: int) -> int:
    # odd seq*2+1 = producer writing
    # even seq*2+2 = stable
    return int(seq) * 2 + 2


def begin_write_slot(buf: memoryview, seq: int, num_slots: int) -> int:
    slot = int(seq) % int(num_slots)
    write_i64(buf, gen_offset(slot), int(seq) * 2 + 1)
    return slot


def publish_slot(buf: memoryview, seq: int, slot: int) -> None:
    write_i64(buf, gen_offset(slot), expected_generation(seq))
    write_i64(buf, LATEST_OFFSET, int(seq))


def read_latest(buf: memoryview) -> int:
    return read_i64(buf, LATEST_OFFSET)


def read_stop(buf: memoryview) -> int:
    return read_i64(buf, STOP_OFFSET)


def is_slot_stable_for_seq(buf: memoryview, seq: int, num_slots: int) -> bool:
    if seq < 0:
        return False
    slot = int(seq) % int(num_slots)
    g = read_i64(buf, gen_offset(slot))
    return g == expected_generation(seq) and (g & 1) == 0

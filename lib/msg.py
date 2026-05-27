from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass

IOX2_MAGIC = b"CIPCIOX2"
IOX2_VERSION = 1
CUDA_IPC_HANDLE_BYTES = 64


class Iox2CUDAIPCStreamInfo(ctypes.Structure):
    """Fixed-size iceoryx2 payload for CUDA IPC stream discovery."""

    _fields_ = [
        ("magic", ctypes.c_char * 8),
        ("version", ctypes.c_uint32),
        ("stream_name", ctypes.c_char * 32),
        ("layout", ctypes.c_char * 32),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("channels", ctypes.c_uint32),
        ("num_slots", ctypes.c_uint32),
        ("itemsize", ctypes.c_uint64),
        ("frame_bytes", ctypes.c_uint64),
        ("handle_len", ctypes.c_uint64),
        ("producer_pid", ctypes.c_uint64),
        ("cuda_ipc_handle", ctypes.c_uint8 * CUDA_IPC_HANDLE_BYTES),
    ]

    @staticmethod
    def type_name() -> str:
        return "CudaIpcStreamInfoV1"

    def __str__(self) -> str:
        return (
            f"Iox2StreamInfo(stream={decode_fixed_bytes(self.stream_name)!r}, "
            f"layout={decode_fixed_bytes(self.layout)!r}, "
            f"shape=({self.num_slots},{self.channels},{self.height},{self.width}), "
            f"frame_bytes={self.frame_bytes}, handle_len={self.handle_len}, "
            f"producer_pid={self.producer_pid})"
        )


class Iox2FrameCHWReady(ctypes.Structure):
    """Fixed-size iceoryx2 payload announcing one completed chw frame."""

    _fields_ = [
        ("magic", ctypes.c_char * 8),
        ("version", ctypes.c_uint32),
        ("stream_name", ctypes.c_char * 32),
        ("seq", ctypes.c_int64),
        ("slot", ctypes.c_uint32),
        ("generation", ctypes.c_int64),
        ("timestamp_ns", ctypes.c_uint64),
    ]

    @staticmethod
    def type_name() -> str:
        return "CudaIpcFrameReadyV1"

    def __str__(self) -> str:
        return (
            f"Iox2FrameCHWReady(stream={decode_fixed_bytes(self.stream_name)!r}, "
            f"seq={self.seq}, slot={self.slot}, generation={self.generation}, "
            f"timestamp_ns={self.timestamp_ns})"
        )


@dataclass(frozen=True)
class StreamInfo:
    stream_name: str
    layout: str
    width: int
    height: int
    channels: int
    num_slots: int
    itemsize: int
    frame_bytes: int
    cuda_ipc_handle: bytes
    producer_pid: int

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self.num_slots, self.channels, self.height, self.width)


@dataclass(frozen=True)
class FrameReady:
    stream_name: str
    seq: int
    slot: int
    generation: int
    timestamp_ns: int


def encode_fixed_bytes(text: str, size: int) -> bytes:
    raw = text.encode("utf-8")
    if len(raw) >= size:
        raw = raw[: size - 1]
    return raw + b"\x00" * (size - len(raw))


def decode_fixed_bytes(raw) -> str:
    return bytes(raw).split(b"\x00", 1)[0].decode("utf-8", errors="replace")


def expected_generation(seq: int) -> int:
    return int(seq) * 2 + 2


def make_stream_info_payload(info: StreamInfo) -> Iox2CUDAIPCStreamInfo:
    handle = bytes(info.cuda_ipc_handle)
    if len(handle) > CUDA_IPC_HANDLE_BYTES:
        raise RuntimeError(f"CUDA IPC handle too large: {len(handle)} > {CUDA_IPC_HANDLE_BYTES}")

    obj = Iox2CUDAIPCStreamInfo()
    obj.magic = IOX2_MAGIC
    obj.version = IOX2_VERSION
    obj.stream_name = encode_fixed_bytes(info.stream_name, 32)
    obj.layout = encode_fixed_bytes(info.layout, 32)
    obj.width = int(info.width)
    obj.height = int(info.height)
    obj.channels = int(info.channels)
    obj.num_slots = int(info.num_slots)
    obj.itemsize = int(info.itemsize)
    obj.frame_bytes = int(info.frame_bytes)
    obj.handle_len = int(len(handle))
    obj.producer_pid = int(info.producer_pid)

    padded = handle + b"\x00" * (CUDA_IPC_HANDLE_BYTES - len(handle))
    obj.cuda_ipc_handle = (ctypes.c_uint8 * CUDA_IPC_HANDLE_BYTES)(*padded)
    return obj


def parse_stream_info_payload(obj: Iox2CUDAIPCStreamInfo) -> StreamInfo:
    if bytes(obj.magic) != IOX2_MAGIC:
        raise RuntimeError(f"bad StreamInfo magic: {bytes(obj.magic)!r}")
    if int(obj.version) != IOX2_VERSION:
        raise RuntimeError(f"bad StreamInfo version: {obj.version}")

    handle_len = int(obj.handle_len)
    if handle_len <= 0 or handle_len > CUDA_IPC_HANDLE_BYTES:
        raise RuntimeError(f"invalid CUDA IPC handle length: {handle_len}")

    handle = bytes(bytearray(obj.cuda_ipc_handle)[:handle_len])

    return StreamInfo(
        stream_name=decode_fixed_bytes(obj.stream_name),
        layout=decode_fixed_bytes(obj.layout),
        width=int(obj.width),
        height=int(obj.height),
        channels=int(obj.channels),
        num_slots=int(obj.num_slots),
        itemsize=int(obj.itemsize),
        frame_bytes=int(obj.frame_bytes),
        cuda_ipc_handle=handle,
        producer_pid=int(obj.producer_pid),
    )


def make_frame_ready_payload(frame: FrameReady) -> Iox2FrameCHWReady:
    obj = Iox2FrameCHWReady()
    obj.magic = IOX2_MAGIC
    obj.version = IOX2_VERSION
    obj.stream_name = encode_fixed_bytes(frame.stream_name, 32)
    obj.seq = int(frame.seq)
    obj.slot = int(frame.slot)
    obj.generation = int(frame.generation)
    obj.timestamp_ns = int(frame.timestamp_ns)
    return obj


def parse_frame_ready_payload(obj: Iox2FrameCHWReady) -> FrameReady:
    if bytes(obj.magic) != IOX2_MAGIC:
        raise RuntimeError(f"bad FrameReady magic: {bytes(obj.magic)!r}")
    if int(obj.version) != IOX2_VERSION:
        raise RuntimeError(f"bad FrameReady version: {obj.version}")

    return FrameReady(
        stream_name=decode_fixed_bytes(obj.stream_name),
        seq=int(obj.seq),
        slot=int(obj.slot),
        generation=int(obj.generation),
        timestamp_ns=int(obj.timestamp_ns),
    )


def service_base_name(prefix: str, stream_name: str) -> str:
    prefix = prefix.strip("/").replace("\\", "/")
    stream = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in stream_name)
    return f"{prefix}/{stream}"


def stream_info_service_name(prefix: str, stream_name: str) -> str:
    return f"{service_base_name(prefix, stream_name)}/StreamInfo"


def frame_ready_service_name(prefix: str, stream_name: str) -> str:
    return f"{service_base_name(prefix, stream_name)}/FrameReady"


def make_stream_info_payload(info: StreamInfo) -> Iox2CUDAIPCStreamInfo:
    handle = bytes(info.cuda_ipc_handle)
    if len(handle) > CUDA_IPC_HANDLE_BYTES:
        raise RuntimeError(f"CUDA IPC handle too large: {len(handle)} > {CUDA_IPC_HANDLE_BYTES}")

    obj = Iox2CUDAIPCStreamInfo()
    obj.magic = IOX2_MAGIC
    obj.version = IOX2_VERSION
    obj.stream_name = encode_fixed_bytes(info.stream_name, 32)
    obj.layout = encode_fixed_bytes(info.layout, 32)
    obj.width = int(info.width)
    obj.height = int(info.height)
    obj.channels = int(info.channels)
    obj.num_slots = int(info.num_slots)
    obj.itemsize = int(info.itemsize)
    obj.frame_bytes = int(info.frame_bytes)
    obj.handle_len = int(len(handle))
    obj.producer_pid = int(info.producer_pid)

    padded = handle + b"\x00" * (CUDA_IPC_HANDLE_BYTES - len(handle))
    obj.cuda_ipc_handle = (ctypes.c_uint8 * CUDA_IPC_HANDLE_BYTES)(*padded)
    return obj


def make_frame_ready_payload(frame: FrameReady) -> Iox2FrameCHWReady:
    obj = Iox2FrameCHWReady()
    obj.magic = IOX2_MAGIC
    obj.version = IOX2_VERSION
    obj.stream_name = encode_fixed_bytes(frame.stream_name, 32)
    obj.seq = int(frame.seq)
    obj.slot = int(frame.slot)
    obj.generation = int(frame.generation)
    obj.timestamp_ns = int(frame.timestamp_ns)
    return obj


def parse_stream_info_payload(obj: Iox2CUDAIPCStreamInfo) -> StreamInfo:
    if bytes(obj.magic) != IOX2_MAGIC:
        raise RuntimeError(f"bad StreamInfo magic: {bytes(obj.magic)!r}")
    if int(obj.version) != IOX2_VERSION:
        raise RuntimeError(f"bad StreamInfo version: {obj.version}")

    handle_len = int(obj.handle_len)
    if handle_len <= 0 or handle_len > CUDA_IPC_HANDLE_BYTES:
        raise RuntimeError(f"invalid CUDA IPC handle length: {handle_len}")

    handle = bytes(bytearray(obj.cuda_ipc_handle)[:handle_len])

    return StreamInfo(
        stream_name=decode_fixed_bytes(obj.stream_name),
        layout=decode_fixed_bytes(obj.layout),
        width=int(obj.width),
        height=int(obj.height),
        channels=int(obj.channels),
        num_slots=int(obj.num_slots),
        itemsize=int(obj.itemsize),
        frame_bytes=int(obj.frame_bytes),
        cuda_ipc_handle=handle,
        producer_pid=int(obj.producer_pid),
    )


def parse_frame_ready_payload(obj: Iox2FrameCHWReady) -> FrameReady:
    if bytes(obj.magic) != IOX2_MAGIC:
        raise RuntimeError(f"bad FrameReady magic: {bytes(obj.magic)!r}")
    if int(obj.version) != IOX2_VERSION:
        raise RuntimeError(f"bad FrameReady version: {obj.version}")

    return FrameReady(
        stream_name=decode_fixed_bytes(obj.stream_name),
        seq=int(obj.seq),
        slot=int(obj.slot),
        generation=int(obj.generation),
        timestamp_ns=int(obj.timestamp_ns),
    )

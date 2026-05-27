from __future__ import annotations

import os
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory

import cupy as cp
import torch

from .shm_layout import (
    DEFAULT_SHM_SIZE,
    DTYPE_UINT8,
    LAYOUT_GRAY_CHW,
    LAYOUT_RGBA_CHW,
    LAYOUT_RGBP_CHW,
    StreamHeader,
    begin_write_slot,
    create_clean_shm,
    initialize_control,
    mark_stopped,
    publish_slot,
    shm_name_for,
    write_handle_bytes,
    write_header,
)
from .torch_cupy import decoded_tensor_to_chw_uint8, torch_tensor_to_cupy_view


def layout_for_channels(channels: int) -> str:
    if channels == 1:
        return LAYOUT_GRAY_CHW
    if channels == 3:
        return LAYOUT_RGBP_CHW
    if channels == 4:
        return LAYOUT_RGBA_CHW
    raise RuntimeError(f"unsupported channel count for IPC layout: {channels}")


@dataclass
class PublishStats:
    seq: int = 0
    published: int = 0


class CudaIpcRingPublisher:
    """
    Owns a persistent CuPy GPU ring buffer and publishes it through CUDA IPC.

    The producer owns this memory for the whole process lifetime. Consumers should
    only read from it. This avoids exporting decoder-owned/recycled surfaces.
    """

    def __init__(
        self,
        stream_name: str,
        first_frame_chw: torch.Tensor,
        num_slots: int = 4,
        gpu_id: int = 0,
        shm_prefix: str = "cuda_ipc_stream",
        shm_size: int = DEFAULT_SHM_SIZE,
    ):
        cp.cuda.Device(gpu_id).use()
        torch.cuda.set_device(gpu_id)

        first_frame_chw = decoded_tensor_to_chw_uint8(first_frame_chw)

        self.stream_name = stream_name
        self.num_slots = int(num_slots)
        self.gpu_id = int(gpu_id)
        self.shm_name = shm_name_for(stream_name, prefix=shm_prefix)
        self.shm_size = int(shm_size)

        self.channels = int(first_frame_chw.shape[0])
        self.height = int(first_frame_chw.shape[1])
        self.width = int(first_frame_chw.shape[2])
        self.itemsize = 1
        self.frame_bytes = self.channels * self.height * self.width * self.itemsize
        self.layout = layout_for_channels(self.channels)

        self.frames = cp.empty(
            (self.num_slots, self.channels, self.height, self.width),
            dtype=cp.uint8,
        )

        handle = cp.cuda.runtime.ipcGetMemHandle(self.frames.data.ptr)

        self.shm: SharedMemory = create_clean_shm(self.shm_name, self.shm_size)
        self.buf = self.shm.buf

        initialize_control(self.buf, self.num_slots)
        write_handle_bytes(self.buf, handle)

        header = StreamHeader(
            stream_name=self.stream_name,
            layout=self.layout,
            width=self.width,
            height=self.height,
            channels=self.channels,
            num_slots=self.num_slots,
            itemsize=self.itemsize,
            frame_bytes=self.frame_bytes,
            handle_len=len(handle),
            producer_pid=os.getpid(),
        )
        write_header(self.buf, header)

        self.stats = PublishStats()

        print(
            f"[publisher:{self.stream_name}] shm={self.shm_name} "
            f"shape={self.frames.shape} layout={self.layout} "
            f"gpu_ptr={hex(self.frames.data.ptr)} handle_bytes={len(handle)}",
            flush=True,
        )

        # Publish the first frame immediately.
        self.publish(first_frame_chw)

    def _check_shape(self, frame_chw: torch.Tensor) -> torch.Tensor:
        frame_chw = decoded_tensor_to_chw_uint8(frame_chw)
        expected = (self.channels, self.height, self.width)
        got = tuple(frame_chw.shape)
        if got != expected:
            raise RuntimeError(
                f"[{self.stream_name}] shape changed. expected={expected}, got={got}. "
                "For now, restart publisher if stream resolution/layout changes."
            )
        return frame_chw

    def publish(self, frame_chw: torch.Tensor) -> int:
        frame_chw = self._check_shape(frame_chw)

        seq = self.stats.seq
        slot = begin_write_slot(self.buf, seq, self.num_slots)

        # Conservative sync: make sure decoder/torch writes are complete before CuPy reads.
        # For higher performance, replace this with CUDA event/stream handoff.
        torch.cuda.synchronize(device=self.gpu_id)

        cp_src = torch_tensor_to_cupy_view(frame_chw)
        cp.copyto(self.frames[slot], cp_src)

        # Publish only after GPU copy is complete.
        cp.cuda.runtime.deviceSynchronize()

        publish_slot(self.buf, seq, slot)

        self.stats.seq += 1
        self.stats.published += 1
        return seq

    def close(self) -> None:
        try:
            mark_stopped(self.buf)
        except Exception:
            pass

        try:
            self.shm.close()
        except Exception:
            pass

        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

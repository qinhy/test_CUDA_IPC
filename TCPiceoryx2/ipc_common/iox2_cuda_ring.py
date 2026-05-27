from __future__ import annotations

import os
import time
from dataclasses import dataclass

import cupy as cp
import torch

from .cuda_ipc_ring import layout_for_channels
from .iox2_messages import FrameReady, StreamInfo, expected_generation, now_ns
from .iox2_transport import Iox2PublisherTransport
from .torch_cupy import decoded_tensor_to_chw_uint8, torch_tensor_to_cupy_view


@dataclass
class Iox2PublishStats:
    seq: int = 0
    published: int = 0


class Iox2CudaIpcRingPublisher:
    """Persistent GPU ring buffer with iceoryx2 metadata/pub-sub control plane."""

    def __init__(
        self,
        stream_name: str,
        first_frame_chw: torch.Tensor,
        num_slots: int = 16,
        gpu_id: int = 0,
        service_prefix: str = "CudaIpcVideo",
        stream_info_period_sec: float = 1.0,
    ):
        cp.cuda.Device(gpu_id).use()
        torch.cuda.set_device(gpu_id)

        first_frame_chw = decoded_tensor_to_chw_uint8(first_frame_chw)

        self.stream_name = stream_name
        self.num_slots = int(num_slots)
        self.gpu_id = int(gpu_id)
        self.service_prefix = service_prefix
        self.stream_info_period_sec = float(stream_info_period_sec)
        self.last_info_time = 0.0

        self.channels = int(first_frame_chw.shape[0])
        self.height = int(first_frame_chw.shape[1])
        self.width = int(first_frame_chw.shape[2])
        self.itemsize = 1
        self.frame_bytes = self.channels * self.height * self.width
        self.layout = layout_for_channels(self.channels)

        self.frames = cp.empty((self.num_slots, self.channels, self.height, self.width), dtype=cp.uint8)
        self.handle = bytes(cp.cuda.runtime.ipcGetMemHandle(self.frames.data.ptr))

        self.transport = Iox2PublisherTransport(stream_name=self.stream_name, service_prefix=self.service_prefix)
        self.info = StreamInfo(
            stream_name=self.stream_name,
            layout=self.layout,
            width=self.width,
            height=self.height,
            channels=self.channels,
            num_slots=self.num_slots,
            itemsize=self.itemsize,
            frame_bytes=self.frame_bytes,
            cuda_ipc_handle=self.handle,
            producer_pid=os.getpid(),
        )

        print(
            f"[iox2-ring:{self.stream_name}] shape={self.frames.shape} layout={self.layout} "
            f"gpu_ptr={hex(self.frames.data.ptr)} handle_bytes={len(self.handle)}",
            flush=True,
        )

        self.stats = Iox2PublishStats()
        self.publish_stream_info(force=True)
        self.publish(first_frame_chw)

    def publish_stream_info(self, force: bool = False) -> None:
        now = time.time()
        if force or (now - self.last_info_time) >= self.stream_info_period_sec:
            self.transport.publish_stream_info(self.info)
            self.last_info_time = now

    def _check_shape(self, frame_chw: torch.Tensor) -> torch.Tensor:
        frame_chw = decoded_tensor_to_chw_uint8(frame_chw)
        expected = (self.channels, self.height, self.width)
        got = tuple(frame_chw.shape)
        if got != expected:
            raise RuntimeError(f"[{self.stream_name}] shape changed. expected={expected}, got={got}")
        return frame_chw

    def publish(self, frame_chw: torch.Tensor) -> int:
        frame_chw = self._check_shape(frame_chw)
        seq = self.stats.seq
        slot = seq % self.num_slots

        torch.cuda.synchronize(device=self.gpu_id)
        cp_src = torch_tensor_to_cupy_view(frame_chw)
        cp.copyto(self.frames[slot], cp_src)
        cp.cuda.runtime.deviceSynchronize()

        msg = FrameReady(
            stream_name=self.stream_name,
            seq=seq,
            slot=slot,
            generation=expected_generation(seq),
            timestamp_ns=now_ns(),
        )
        self.transport.publish_frame_ready(msg)
        self.publish_stream_info(force=False)

        self.stats.seq += 1
        self.stats.published += 1
        return seq

    def close(self) -> None:
        pass

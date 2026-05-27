from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import cupy as cp
import torch
import iceoryx2 as iox2

try:
    from .utils import decoded_tensor_to_chw_uint8, layout_for_channels, torch_tensor_to_cupy_view, now_ns
    from .msg import (
        FrameReady,
        Iox2CUDAIPCStreamInfo,
        Iox2FrameCHWReady,
        StreamInfo,
        
        expected_generation,
        frame_ready_service_name,
        stream_info_service_name,

        make_frame_ready_payload,
        make_stream_info_payload,
        parse_frame_ready_payload,
        parse_stream_info_payload,
    )
except:
    from utils import decoded_tensor_to_chw_uint8, layout_for_channels, torch_tensor_to_cupy_view, now_ns
    from msg import (
        FrameReady,
        Iox2CUDAIPCStreamInfo,
        Iox2FrameCHWReady,
        StreamInfo,
        
        expected_generation,
        frame_ready_service_name,
        stream_info_service_name,

        make_frame_ready_payload,
        make_stream_info_payload,
        parse_frame_ready_payload,
        parse_stream_info_payload,
    )

class Iox2PublisherTransport:
    """iceoryx2 publisher for CUDA IPC StreamInfo and FrameReady messages."""

    def __init__(self, stream_name: str, service_prefix: str = "CudaIpcVideo"):
        self.stream_name = stream_name
        self.service_prefix = service_prefix

        try:
            iox2.set_log_level_from_env_or(iox2.LogLevel.Info)
        except Exception:
            pass

        self.node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
        self.info_service_name = stream_info_service_name(service_prefix, stream_name)
        self.frame_service_name = frame_ready_service_name(service_prefix, stream_name)

        self.info_service = (
            self.node.service_builder(iox2.ServiceName.new(self.info_service_name))
            .publish_subscribe(Iox2CUDAIPCStreamInfo)
            .open_or_create()
        )
        self.frame_service = (
            self.node.service_builder(iox2.ServiceName.new(self.frame_service_name))
            .publish_subscribe(Iox2FrameCHWReady)
            .open_or_create()
        )

        self.info_pub = self.info_service.publisher_builder().create()
        self.frame_pub = self.frame_service.publisher_builder().create()

        print(
            f"[iox2-pub:{stream_name}] info={self.info_service_name} frames={self.frame_service_name}",
            flush=True,
        )

    def publish_stream_info(self, info: StreamInfo) -> None:
        sample = self.info_pub.loan_uninit()
        sample = sample.write_payload(make_stream_info_payload(info))
        sample.send()

    def publish_frame_ready(self, frame: FrameReady) -> None:
        sample = self.frame_pub.loan_uninit()
        sample = sample.write_payload(make_frame_ready_payload(frame))
        sample.send()


class Iox2SubscriberTransport:
    """iceoryx2 subscriber for CUDA IPC StreamInfo and FrameReady messages."""

    def __init__(self, stream_name: str, service_prefix: str = "CudaIpcVideo"):
        self.stream_name = stream_name
        self.service_prefix = service_prefix

        try:
            iox2.set_log_level_from_env_or(iox2.LogLevel.Info)
        except Exception:
            pass

        self.node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
        self.info_service_name = stream_info_service_name(service_prefix, stream_name)
        self.frame_service_name = frame_ready_service_name(service_prefix, stream_name)

        self.info_service = (
            self.node.service_builder(iox2.ServiceName.new(self.info_service_name))
            .publish_subscribe(Iox2CUDAIPCStreamInfo)
            .open_or_create()
        )
        self.frame_service = (
            self.node.service_builder(iox2.ServiceName.new(self.frame_service_name))
            .publish_subscribe(Iox2FrameCHWReady)
            .open_or_create()
        )

        self.info_sub = self.info_service.subscriber_builder().create()
        self.frame_sub = self.frame_service.subscriber_builder().create()

        print(
            f"[iox2-sub:{stream_name}] info={self.info_service_name} frames={self.frame_service_name}",
            flush=True,
        )

    def receive_stream_info_once(self) -> Optional[StreamInfo]:
        sample = self.info_sub.receive()
        if sample is None:
            return None
        return parse_stream_info_payload(sample.payload().contents)

    def wait_stream_info(self, timeout_sec: float = 30.0, poll_sec: float = 0.01) -> StreamInfo:
        deadline = time.time() + timeout_sec
        last_err: Optional[BaseException] = None
        while time.time() < deadline:
            try:
                info = self.receive_stream_info_once()
                if info is not None:
                    return info
            except Exception as e:
                last_err = e
            time.sleep(poll_sec)
        raise TimeoutError(f"timed out waiting for iceoryx2 StreamInfo: {last_err}")

    def receive_frame_ready_once(self) -> Optional[FrameReady]:
        sample = self.frame_sub.receive()
        if sample is None:
            return None
        return parse_frame_ready_payload(sample.payload().contents)

    def drain_frame_ready(self) -> list[FrameReady]:
        out: list[FrameReady] = []
        while True:
            msg = self.receive_frame_ready_once()
            if msg is None:
                break
            out.append(msg)
        return out

    def receive_latest_frame_ready(self) -> Optional[FrameReady]:
        msgs = self.drain_frame_ready()
        return msgs[-1] if msgs else None


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

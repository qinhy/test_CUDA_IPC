from __future__ import annotations

import time
from typing import Optional

from .iox2_messages import (
    FrameReady,
    Iox2FrameReady,
    Iox2StreamInfo,
    StreamInfo,
    frame_ready_service_name,
    make_frame_ready_payload,
    make_stream_info_payload,
    parse_frame_ready_payload,
    parse_stream_info_payload,
    stream_info_service_name,
)


def require_iox2():
    try:
        import iceoryx2 as iox2
        return iox2
    except ImportError as e:
        raise RuntimeError("iceoryx2 is not installed. Try: uv add iceoryx2") from e


class Iox2PublisherTransport:
    """iceoryx2 publisher for CUDA IPC StreamInfo and FrameReady messages."""

    def __init__(self, stream_name: str, service_prefix: str = "CudaIpcVideo"):
        self.stream_name = stream_name
        self.service_prefix = service_prefix
        self.iox2 = require_iox2()

        try:
            self.iox2.set_log_level_from_env_or(self.iox2.LogLevel.Info)
        except Exception:
            pass

        self.node = self.iox2.NodeBuilder.new().create(self.iox2.ServiceType.Ipc)
        self.info_service_name = stream_info_service_name(service_prefix, stream_name)
        self.frame_service_name = frame_ready_service_name(service_prefix, stream_name)

        self.info_service = (
            self.node.service_builder(self.iox2.ServiceName.new(self.info_service_name))
            .publish_subscribe(Iox2StreamInfo)
            .open_or_create()
        )
        self.frame_service = (
            self.node.service_builder(self.iox2.ServiceName.new(self.frame_service_name))
            .publish_subscribe(Iox2FrameReady)
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
        self.iox2 = require_iox2()

        try:
            self.iox2.set_log_level_from_env_or(self.iox2.LogLevel.Info)
        except Exception:
            pass

        self.node = self.iox2.NodeBuilder.new().create(self.iox2.ServiceType.Ipc)
        self.info_service_name = stream_info_service_name(service_prefix, stream_name)
        self.frame_service_name = frame_ready_service_name(service_prefix, stream_name)

        self.info_service = (
            self.node.service_builder(self.iox2.ServiceName.new(self.info_service_name))
            .publish_subscribe(Iox2StreamInfo)
            .open_or_create()
        )
        self.frame_service = (
            self.node.service_builder(self.iox2.ServiceName.new(self.frame_service_name))
            .publish_subscribe(Iox2FrameReady)
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

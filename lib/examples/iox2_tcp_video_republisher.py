#!/usr/bin/env python3
"""
iox2_tcp_video_republisher.py

Republish encoded TCP video byte streams into iceoryx2 publish/subscribe services.

Designed for tcp_sim_src.py:
  rgb   HEVC/H.265  tcp://<host>:5000 -> iceoryx2 service sim/video/rgb
  left  H.264       tcp://<host>:5001 -> iceoryx2 service sim/video/left
  right H.264       tcp://<host>:5002 -> iceoryx2 service sim/video/right

The payload is a dynamic iceoryx2 Slice[ctypes.c_uint8]. Each sample is one TCP
read chunk, not necessarily one video frame or one complete NAL unit. This is
appropriate for byte-stream decoders such as ffmpeg/ffplay/gstreamer that accept
Annex-B H.264/H.265 byte streams.
"""

from __future__ import annotations

import argparse
import ctypes
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass
from typing import Iterable

try:
    import iceoryx2 as iox2
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import iceoryx2. Install the Python bindings first, for example\n"
        "  python3 -m pip install iceoryx2\n"
        "or build/install them from the eclipse-iceoryx/iceoryx2 repository."
    ) from exc


@dataclass(frozen=True)
class StreamSpec:
    name: str
    port: int
    codec: str


STREAMS: dict[str, StreamSpec] = {
    "rgb": StreamSpec("rgb", 5000, "hevc"),
    "left": StreamSpec("left", 5001, "h264"),
    "right": StreamSpec("right", 5002, "h264"),
}


def parse_streams(value: str) -> list[StreamSpec]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("at least one stream is required")

    specs: list[StreamSpec] = []
    for name in names:
        try:
            specs.append(STREAMS[name])
        except KeyError as exc:
            valid = ",".join(STREAMS)
            raise argparse.ArgumentTypeError(f"unknown stream {name!r}; valid streams: {valid}") from exc
    return specs


def make_service_name(prefix: str, stream_name: str) -> str:
    return f"{prefix.rstrip('/')}/{stream_name}"


class Iox2BytesPublisher:
    """Thin wrapper around an iceoryx2 Slice[uint8] publisher."""

    def __init__(self, node, service_name: str, initial_max_slice_len: int):
        self.service_name = service_name
        service = (
            node.service_builder(iox2.ServiceName.new(service_name))
            .publish_subscribe(iox2.Slice[ctypes.c_uint8])
            .open_or_create()
        )
        self.publisher = (
            service.publisher_builder()
            .initial_max_slice_len(initial_max_slice_len)
            .allocation_strategy(iox2.AllocationStrategy.PowerOfTwo)
            .create()
        )

    def publish(self, payload: bytes) -> None:
        if not payload:
            return

        sample = self.publisher.loan_slice_uninit(len(payload))
        dst = sample.payload()
        for idx, value in enumerate(payload):
            dst[idx] = value

        sample.assume_init().send()


def tcp_to_iox2_worker(
    *,
    node,
    tcp_host: str,
    spec: StreamSpec,
    service_prefix: str,
    chunk_size: int,
    reconnect_delay: float,
    stop_event: threading.Event,
) -> None:
    service_name = make_service_name(service_prefix, spec.name)
    publisher = Iox2BytesPublisher(node, service_name, initial_max_slice_len=chunk_size)

    print(
        f"[{spec.name}] publishing tcp://{tcp_host}:{spec.port} "
        f"({spec.codec}) -> iceoryx2 service {service_name!r}",
        flush=True,
    )

    while not stop_event.is_set():
        sock: socket.socket | None = None
        try:
            sock = socket.create_connection((tcp_host, spec.port), timeout=3.0)
            sock.settimeout(1.0)
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass

            print(f"[{spec.name}] connected", flush=True)
            byte_count = 0
            t0 = time.perf_counter()

            while not stop_event.is_set():
                try:
                    data = sock.recv(chunk_size)
                except socket.timeout:
                    continue

                if not data:
                    raise ConnectionError("TCP source closed the connection")

                publisher.publish(data)
                byte_count += len(data)

                now = time.perf_counter()
                if now - t0 >= 5.0:
                    mbps = (byte_count * 8.0) / (now - t0) / 1_000_000.0
                    print(f"[{spec.name}] republish {mbps:.2f} Mbit/s", flush=True)
                    byte_count = 0
                    t0 = now

        except Exception as exc:
            if not stop_event.is_set():
                print(f"[{spec.name}] disconnected: {exc}; reconnecting", file=sys.stderr, flush=True)
                stop_event.wait(reconnect_delay)
        finally:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

    print(f"[{spec.name}] stopped", flush=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Republish encoded TCP video byte streams into iceoryx2 pub/sub services."
    )
    parser.add_argument("--tcp-host", default="127.0.0.1", help="TCP source host/IP")
    parser.add_argument("--streams", type=parse_streams, default=parse_streams("rgb"), help="Comma-separated streams: rgb,left,right")
    parser.add_argument("--service-prefix", default="sim/video", help="iceoryx2 service prefix")
    parser.add_argument("--chunk-size", type=int, default=64 * 1024, help="Max bytes per iceoryx2 sample")
    parser.add_argument("--reconnect-delay", type=float, default=1.0, help="Seconds before reconnect after disconnect")
    parser.add_argument("--log-level", default="Info", choices=["Trace", "Debug", "Info", "Warn", "Error", "Fatal"])
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    iox2.set_log_level_from_env_or(getattr(iox2.LogLevel, args.log_level))
    node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)

    stop_event = threading.Event()

    def request_stop(signum, _frame):
        print(f"\nreceived signal {signum}; stopping", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    threads: list[threading.Thread] = []
    for spec in args.streams:
        thread = threading.Thread(
            target=tcp_to_iox2_worker,
            kwargs={
                "node": node,
                "tcp_host": args.tcp_host,
                "spec": spec,
                "service_prefix": args.service_prefix,
                "chunk_size": args.chunk_size,
                "reconnect_delay": args.reconnect_delay,
                "stop_event": stop_event,
            },
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    try:
        while not stop_event.is_set():
            try:
                node.wait(iox2.Duration.from_millis(200))
            except iox2.NodeWaitFailure:
                stop_event.set()
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=2.0)

    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

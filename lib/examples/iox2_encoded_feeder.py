#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import threading
import time
from typing import Any

import iceoryx2 as iox2


def _payload_to_bytes(payload: Any) -> bytes:
    """Copy an iceoryx2 dynamic uint8 slice payload into Python bytes.

    The Python iceoryx2 Slice object intentionally does not behave exactly like a
    Python bytes-like object in every release.  The safe fallback is indexed copy.
    """
    n = payload.len()
    if n <= 0:
        return b""

    # Some versions expose an iterable/buffer-like wrapper. Try it first.
    try:
        out = bytes(payload)
        if len(out) == n:
            return out
    except Exception:
        pass

    return bytes(int(payload[i]) & 0xFF for i in range(n))


class Iox2EncodedByteFeeder:
    """Blocking byte feeder for PyNvVideoCodec.CreateDemuxer.

    It subscribes to an iceoryx2 service that publishes `iox2.Slice[ctypes.c_uint8]`
    samples, copies received samples into an internal byte buffer, and returns a
    byte chunk whenever PyNvVideoCodec asks for more input.

    This is the iceoryx2 replacement for a TCP `SocketFeeder` whose `feed_chunk()`
    callback returned bytes from `socket.recv(...)`.
    """

    def __init__(
        self,
        service_name: str,
        stop_event: threading.Event,
        *,
        max_return_bytes: int = 256 * 1024,
        poll_sec: float = 0.001,
        drain_limit: int = 64,
        verbose: bool = False,
    ) -> None:
        self.service_name = service_name
        self.stop_event = stop_event
        self.max_return_bytes = max(1, int(max_return_bytes))
        self.poll_sec = max(0.0, float(poll_sec))
        self.drain_limit = max(1, int(drain_limit))
        self.verbose = verbose
        self._closed = False
        self._buffer = bytearray()
        self._samples = 0
        self._bytes = 0

        # Keep these objects alive for the whole feeder lifetime.
        self.node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
        self.service = (
            self.node.service_builder(iox2.ServiceName.new(service_name))
            .publish_subscribe(iox2.Slice[ctypes.c_uint8])
            .open_or_create()
        )
        self.subscriber = self.service.subscriber_builder().create()

        if self.verbose:
            print(f"[iox2-in] subscribed to '{service_name}'", flush=True)

    def close(self) -> None:
        self._closed = True

    def feed_chunk(self, demuxer_buffer: Any | None = None) -> int | bytes:
        """Feed bytes to PyNvVideoCodec.CreateDemuxer.

        PyNvVideoCodec 2.x calls the callback with one pre-allocated writable
        buffer argument and expects the callback to copy bytes into that buffer
        and return the number of copied bytes.

        Older/local feeders sometimes used a no-argument callback that returned
        bytes.  This method supports both forms:

        * feed_chunk(buffer) -> int
        * feed_chunk() -> bytes
        """
        if demuxer_buffer is None:
            return self._read_bytes(self.max_return_bytes)

        try:
            capacity = len(demuxer_buffer)
        except Exception:
            capacity = self.max_return_bytes

        if capacity <= 0:
            return 0

        chunk = self._read_bytes(min(capacity, self.max_return_bytes))
        if not chunk:
            return 0

        n = len(chunk)

        # Fast path for bytearray/memoryview/numpy-like writable buffers.
        try:
            demuxer_buffer[:n] = chunk
            return n
        except Exception:
            pass

        # Conservative fallback for pybind buffers exposing item assignment.
        for i, b in enumerate(chunk):
            demuxer_buffer[i] = b
        return n

    def _read_bytes(self, max_bytes: int) -> bytes:
        """Block until at least one encoded byte is available or EOF/stop."""
        max_bytes = max(1, int(max_bytes))

        while not self._closed and not self.stop_event.is_set():
            if self._buffer:
                n = min(len(self._buffer), max_bytes)
                out = bytes(self._buffer[:n])
                del self._buffer[:n]
                return out

            self._drain_available_samples()
            if self._buffer:
                continue

            if self.poll_sec > 0.0:
                time.sleep(self.poll_sec)

        return b""

    def _drain_available_samples(self) -> None:
        for _ in range(self.drain_limit):
            sample = self.subscriber.receive()
            if sample is None:
                return

            payload = sample.payload()
            chunk = _payload_to_bytes(payload)
            if not chunk:
                continue

            self._buffer.extend(chunk)
            self._samples += 1
            self._bytes += len(chunk)

            if self.verbose and self._samples % 300 == 0:
                mib = self._bytes / (1024.0 * 1024.0)
                print(
                    f"[iox2-in:{self.service_name}] samples={self._samples} bytes={mib:.1f} MiB buffered={len(self._buffer)}",
                    flush=True,
                )

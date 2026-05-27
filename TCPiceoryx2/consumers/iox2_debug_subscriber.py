#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipc_common.iox2_transport import Iox2SubscriberTransport


def main() -> int:
    p = argparse.ArgumentParser(description="Debug iceoryx2 StreamInfo/FrameReady messages")
    p.add_argument("--stream", default="rgb")
    p.add_argument("--iox2-prefix", default="CudaIpcVideo")
    p.add_argument("--timeout", type=float, default=30.0)
    args = p.parse_args()
    sub = Iox2SubscriberTransport(args.stream, service_prefix=args.iox2_prefix)
    info = sub.wait_stream_info(timeout_sec=args.timeout)
    print("[debug-iox2] stream info:", info, flush=True)
    count = 0
    try:
        while True:
            for m in sub.drain_frame_ready():
                count += 1
                if count % 30 == 0:
                    print("[debug-iox2]", m, flush=True)
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

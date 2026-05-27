#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cupy as cp

from ipc_common.shm_layout import (
    LATEST_OFFSET,
    STOP_OFFSET,
    expected_generation,
    gen_offset,
    get_handle_bytes,
    read_i64,
    shm_name_for,
    wait_for_valid_header,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Small CuPy debug consumer for CUDA IPC rings")
    p.add_argument("--stream", default="rgb")
    p.add_argument("--shm-prefix", default="cuda_ipc_stream")
    p.add_argument("--shm-name", default=None)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--timeout", type=float, default=30.0)
    args = p.parse_args()

    cp.cuda.Device(args.gpu_id).use()

    name = args.shm_name or shm_name_for(args.stream, prefix=args.shm_prefix)
    shm, h = wait_for_valid_header(name, timeout_sec=args.timeout)
    buf = shm.buf

    ptr = None
    try:
        handle = get_handle_bytes(buf, h.handle_len)
        ptr = cp.cuda.runtime.ipcOpenMemHandle(handle)

        nbytes = h.num_slots * h.frame_bytes
        owner = object()
        mem = cp.cuda.UnownedMemory(ptr, nbytes, owner)
        frames = cp.ndarray(h.shape, dtype=cp.uint8, memptr=cp.cuda.MemoryPointer(mem, 0))

        print(f"[debug] attached {name}: layout={h.layout} shape={h.shape} ptr={hex(ptr)}", flush=True)

        last = -1
        while True:
            latest = read_i64(buf, LATEST_OFFSET)
            stop = read_i64(buf, STOP_OFFSET)

            if latest > last:
                slot = latest % h.num_slots
                expected = expected_generation(latest)
                g1 = read_i64(buf, gen_offset(slot))

                if g1 == expected:
                    # Tiny CPU copy of scalar stats only.
                    mean = float(frames[slot].mean().get())
                    g2 = read_i64(buf, gen_offset(slot))
                    ok = (g1 == g2)
                    print(f"[debug] seq={latest} slot={slot} mean={mean:.2f} stable={ok}", flush=True)
                    last = latest

            if stop and latest <= last:
                break

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass

    finally:
        if ptr is not None:
            cp.cuda.runtime.ipcCloseMemHandle(ptr)
        try:
            buf.release()
        except Exception:
            pass
        shm.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

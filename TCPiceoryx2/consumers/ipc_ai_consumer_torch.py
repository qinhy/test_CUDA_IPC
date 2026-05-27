#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cupy as cp
import torch

from ipc_common.shm_layout import (
    LATEST_OFFSET,
    STOP_OFFSET,
    StreamHeader,
    expected_generation,
    gen_offset,
    get_handle_bytes,
    read_i64,
    shm_name_for,
    wait_for_valid_header,
)
from ipc_common.torch_cupy import preprocess_for_ai


class DummyAI(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Replace this with your real model.
        return {
            "shape": tuple(x.shape),
            "device": str(x.device),
            "mean": float(x.mean().detach().cpu()),
        }


class IpcTorchConsumer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.shm = None
        self.buf = None
        self.header: StreamHeader | None = None
        self.ptr = None
        self.mem = None
        self.memptr = None
        self.frames = None

        self.last_seen = -1
        self.accepted = 0
        self.dropped = 0

    def attach(self) -> None:
        cp.cuda.Device(self.args.gpu_id).use()
        torch.cuda.set_device(self.args.gpu_id)

        name = self.args.shm_name or shm_name_for(self.args.stream, prefix=self.args.shm_prefix)
        self.shm, self.header = wait_for_valid_header(name, timeout_sec=self.args.timeout)
        self.buf = self.shm.buf

        handle = get_handle_bytes(self.buf, self.header.handle_len)
        self.ptr = cp.cuda.runtime.ipcOpenMemHandle(handle)

        nbytes = self.header.num_slots * self.header.frame_bytes
        owner = object()
        self.mem = cp.cuda.UnownedMemory(self.ptr, nbytes, owner)
        self.memptr = cp.cuda.MemoryPointer(self.mem, 0)
        self.frames = cp.ndarray(
            self.header.shape,
            dtype=cp.uint8,
            memptr=self.memptr,
        )

        print(
            f"[ai] attached {name}: stream={self.header.stream_name} "
            f"layout={self.header.layout} shape={self.header.shape} ptr={hex(self.ptr)}",
            flush=True,
        )

    def get_stable_torch_copy(self, seq: int) -> torch.Tensor | None:
        slot = seq % self.header.num_slots
        expected = expected_generation(seq)

        g1 = read_i64(self.buf, gen_offset(slot))
        if g1 != expected or (g1 & 1):
            self.dropped += 1
            return None

        cp_frame = self.frames[slot]

        # Zero-copy CuPy -> Torch view, then immediately clone on GPU.
        # The clone prevents producer slot reuse from corrupting model input.
        t_view = torch.utils.dlpack.from_dlpack(cp_frame)
        t_copy = t_view.clone()

        torch.cuda.synchronize(device=self.args.gpu_id)

        g2 = read_i64(self.buf, gen_offset(slot))
        if g1 != g2:
            self.dropped += 1
            return None

        return t_copy

    def run(self) -> None:
        self.attach()
        model = DummyAI().cuda().eval()

        last_print = time.time()

        try:
            while True:
                latest = read_i64(self.buf, LATEST_OFFSET)
                stop = read_i64(self.buf, STOP_OFFSET)

                if latest > self.last_seen:
                    if self.args.latest_only:
                        seqs = [latest]
                    else:
                        seqs = range(self.last_seen + 1, latest + 1)

                    for seq in seqs:
                        frame_chw = self.get_stable_torch_copy(seq)
                        self.last_seen = seq

                        if frame_chw is None:
                            continue

                        ai_input = preprocess_for_ai(frame_chw)

                        with torch.inference_mode():
                            result = model(ai_input)

                        self.accepted += 1

                        if self.accepted % self.args.print_every == 0:
                            print(f"[ai] seq={seq} result={result}", flush=True)

                now = time.time()
                if now - last_print >= 1.0:
                    print(
                        f"[ai] latest={latest} seen={self.last_seen} "
                        f"accepted={self.accepted} dropped={self.dropped} stop={stop}",
                        flush=True,
                    )
                    last_print = now

                if stop and latest <= self.last_seen:
                    print("[ai] producer stop detected", flush=True)
                    break

                time.sleep(self.args.poll_sec)

        except KeyboardInterrupt:
            pass

        finally:
            self.close()

    def close(self) -> None:
        try:
            self.frames = None
            self.memptr = None
            self.mem = None
        except Exception:
            pass

        try:
            if self.ptr is not None:
                cp.cuda.runtime.ipcCloseMemHandle(self.ptr)
                self.ptr = None
        except Exception as e:
            print("[ai cleanup] ipcCloseMemHandle:", e, flush=True)

        try:
            if self.buf is not None:
                self.buf.release()
                self.buf = None
        except Exception:
            pass

        try:
            if self.shm is not None:
                self.shm.close()
                self.shm = None
        except Exception as e:
            print("[ai cleanup] shm:", e, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CUDA IPC CHW uint8 ring -> Torch AI consumer")
    p.add_argument("--stream", default="rgb")
    p.add_argument("--shm-prefix", default="cuda_ipc_stream")
    p.add_argument("--shm-name", default=None)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--latest-only", action="store_true", help="skip to latest frame instead of iterating available seqs")
    p.add_argument("--poll-sec", type=float, default=0.005)
    p.add_argument("--print-every", type=int, default=30)
    return p.parse_args()


def main() -> int:
    IpcTorchConsumer(parse_args()).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

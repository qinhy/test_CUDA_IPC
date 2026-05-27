#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cupy as cp
import torch

from ipc_common.iox2_messages import FrameReady, StreamInfo
from ipc_common.iox2_transport import Iox2SubscriberTransport
from ipc_common.torch_cupy import preprocess_for_ai


class DummyAI(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return {"shape": tuple(x.shape), "device": str(x.device), "mean": float(x.mean().detach().cpu())}


class Iox2AiConsumer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.transport: Iox2SubscriberTransport | None = None
        self.info: StreamInfo | None = None
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
        self.transport = Iox2SubscriberTransport(self.args.stream, service_prefix=self.args.iox2_prefix)
        self.info = self.transport.wait_stream_info(timeout_sec=self.args.timeout)
        self.ptr = cp.cuda.runtime.ipcOpenMemHandle(self.info.cuda_ipc_handle)
        nbytes = self.info.num_slots * self.info.frame_bytes
        owner = object()
        self.mem = cp.cuda.UnownedMemory(self.ptr, nbytes, owner)
        self.memptr = cp.cuda.MemoryPointer(self.mem, 0)
        self.frames = cp.ndarray(self.info.shape, dtype=cp.uint8, memptr=self.memptr)
        print(f"[ai] stream={self.info.stream_name} layout={self.info.layout} shape={self.info.shape} ptr={hex(self.ptr)}", flush=True)

    def torch_clone_from_slot(self, slot: int) -> torch.Tensor:
        cp_frame = self.frames[slot]
        # View CUDA IPC memory, then clone on GPU to protect inference from later slot reuse.
        t_view = torch.utils.dlpack.from_dlpack(cp_frame)
        t_copy = t_view.clone()
        torch.cuda.synchronize(device=self.args.gpu_id)
        return t_copy

    def process_message(self, msg: FrameReady, model: torch.nn.Module) -> None:
        if msg.seq <= self.last_seen:
            self.dropped += 1
            return
        if self.args.latest_only and msg.seq > self.last_seen + 1 and self.last_seen >= 0:
            self.dropped += msg.seq - self.last_seen - 1
        frame_chw = self.torch_clone_from_slot(msg.slot)
        ai_input = preprocess_for_ai(frame_chw)
        with torch.inference_mode():
            result = model(ai_input)
        self.last_seen = msg.seq
        self.accepted += 1
        if self.accepted % self.args.print_every == 0:
            print(f"[ai] seq={msg.seq} slot={msg.slot} result={result}", flush=True)

    def run(self) -> None:
        self.attach()
        model = DummyAI().cuda().eval()
        last_print = time.time()
        try:
            while True:
                if self.args.latest_only:
                    msg = self.transport.receive_latest_frame_ready()
                    if msg is not None:
                        self.process_message(msg, model)
                else:
                    for msg in self.transport.drain_frame_ready():
                        self.process_message(msg, model)
                now = time.time()
                if now - last_print >= 1.0:
                    print(f"[ai] seen={self.last_seen} accepted={self.accepted} dropped={self.dropped}", flush=True)
                    last_print = now
                time.sleep(self.args.poll_sec)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        self.frames = None
        self.memptr = None
        self.mem = None
        try:
            if self.ptr is not None:
                cp.cuda.runtime.ipcCloseMemHandle(self.ptr)
                self.ptr = None
        except Exception as e:
            print("[ai cleanup] ipcCloseMemHandle:", e, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="iceoryx2 FrameReady + CUDA IPC -> Torch AI consumer")
    p.add_argument("--stream", default="rgb")
    p.add_argument("--iox2-prefix", default="CudaIpcVideo")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--latest-only", action="store_true")
    p.add_argument("--poll-sec", type=float, default=0.002)
    p.add_argument("--print-every", type=int, default=30)
    return p.parse_args()


def main() -> int:
    Iox2AiConsumer(parse_args()).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

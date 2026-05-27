#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CUDA_ROOT = os.environ.get("CUDA_PATH")
if CUDA_ROOT:
    os.add_dll_directory(os.path.join(CUDA_ROOT, "bin"))

from OpenGL.GL import *
from OpenGL.GLUT import *

import pycuda.driver as cuda
import pycuda.gl as cudagl
from pycuda.compiler import SourceModule

try:
    from .com import Iox2SubscriberTransport
    from .msg import (
        FrameReady,
        StreamInfo,
    )
except:
    from com import Iox2SubscriberTransport
    from msg import (
        FrameReady,
        StreamInfo,        
    )

CUDA_SRC = r"""
extern "C" __global__
void chw_u8_to_rgba_pbo(
    const unsigned char *src,
    unsigned char *dst,
    int width,
    int height,
    int channels,
    int flip_y
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int src_y = flip_y ? (height - 1 - y) : y;
    int pix = src_y * width + x;
    int plane = width * height;

    unsigned char r, g, b, a;
    a = 255;
    if (channels == 1) {
        unsigned char v = src[pix];
        r = v; g = v; b = v;
    } else if (channels == 3) {
        r = src[0 * plane + pix];
        g = src[1 * plane + pix];
        b = src[2 * plane + pix];
    } else if (channels == 4) {
        r = src[0 * plane + pix];
        g = src[1 * plane + pix];
        b = src[2 * plane + pix];
        a = src[3 * plane + pix];
    } else {
        r = 255; g = 0; b = 255; a = 255;
    }

    int dst_i = 4 * (y * width + x);
    dst[dst_i + 0] = r;
    dst[dst_i + 1] = g;
    dst[dst_i + 2] = b;
    dst[dst_i + 3] = a;
}
"""


class Iox2GlViewer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.transport: Iox2SubscriberTransport | None = None
        self.info: StreamInfo | None = None
        self.tex = None
        self.pbo = None
        self.cuda_pbo = None
        self.cuda_ctx = None
        self.ipc_mem = None
        self.remote_base_ptr = None
        self.copy_kernel = None
        self.closing = False
        self.last_displayed_seq = -1
        self.displayed_count = 0
        self.dropped_count = 0
        self.last_status_time = 0.0
        self.pending_latest: FrameReady | None = None

    @property
    def width(self) -> int: return self.info.width
    @property
    def height(self) -> int: return self.info.height
    @property
    def channels(self) -> int: return self.info.channels
    @property
    def frame_bytes(self) -> int: return self.info.frame_bytes

    def attach_iox2_and_wait_info(self) -> None:
        self.transport = Iox2SubscriberTransport(self.args.stream, service_prefix=self.args.iox2_prefix)
        self.info = self.transport.wait_stream_info(timeout_sec=self.args.timeout)
        print(
            f"[viewer] stream info: stream={self.info.stream_name} layout={self.info.layout} "
            f"shape={self.info.shape} frame_bytes={self.info.frame_bytes}",
            flush=True,
        )

    def init_gl_window(self) -> None:
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)
        glutInitWindowSize(self.width, self.height)
        glutCreateWindow(f"iceoryx2 CUDA IPC {self.info.stream_name} -> OpenGL".encode("utf-8"))
        vendor = glGetString(GL_VENDOR)
        renderer = glGetString(GL_RENDERER)
        version = glGetString(GL_VERSION)
        print("[GL] vendor  :", vendor.decode(errors="replace") if vendor else None)
        print("[GL] renderer:", renderer.decode(errors="replace") if renderer else None)
        print("[GL] version :", version.decode(errors="replace") if version else None)
        glViewport(0, 0, self.width, self.height)
        glDisable(GL_DEPTH_TEST)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    def init_gl_resources(self) -> None:
        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        self.pbo = glGenBuffers(1)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo)
        glBufferData(GL_PIXEL_UNPACK_BUFFER, self.width * self.height * 4, None, GL_STREAM_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    def init_cuda_after_gl_context_exists(self) -> None:
        cuda.init()
        self.cuda_ctx = cudagl.make_context(cuda.Device(self.args.gpu_id))
        mod = SourceModule(CUDA_SRC, options=["--use_fast_math"])
        self.copy_kernel = mod.get_function("chw_u8_to_rgba_pbo")
        self.cuda_pbo = cudagl.RegisteredBuffer(int(self.pbo), cudagl.graphics_map_flags.WRITE_DISCARD)

    def open_cuda_ipc_handle(self) -> None:
        self.ipc_mem = cuda.IPCMemoryHandle(bytearray(self.info.cuda_ipc_handle))
        self.remote_base_ptr = int(self.ipc_mem)
        print("[viewer] opened CUDA IPC base ptr:", hex(self.remote_base_ptr), flush=True)

    def update_latest_message(self) -> None:
        msgs = self.transport.drain_frame_ready()
        if not msgs:
            return
        self.pending_latest = msgs[-1]
        if len(msgs) > 1:
            self.dropped_count += len(msgs) - 1

    def copy_frame_to_pbo(self, msg: FrameReady) -> bool:
        if msg.seq == self.last_displayed_seq:
            return False
        src_ptr = self.remote_base_ptr + int(msg.slot) * self.frame_bytes
        mapping = self.cuda_pbo.map()
        try:
            pbo_ptr, pbo_size = mapping.device_ptr_and_size()
            needed = self.width * self.height * 4
            if pbo_size < needed:
                raise RuntimeError(f"PBO too small: {pbo_size} < {needed}")
            block = (16, 16, 1)
            grid = ((self.width + 15) // 16, (self.height + 15) // 16, 1)
            self.copy_kernel(
                np.uintp(src_ptr),
                np.uintp(pbo_ptr),
                np.int32(self.width),
                np.int32(self.height),
                np.int32(self.channels),
                np.int32(1 if self.args.flip_y else 0),
                block=block,
                grid=grid,
            )
            cuda.Context.synchronize()
        finally:
            mapping.unmap()
        self.last_displayed_seq = msg.seq
        self.displayed_count += 1
        return True

    def upload_pbo_to_texture(self) -> None:
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    def draw_fullscreen_quad(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0)
        glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0)
        glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0)
        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0)
        glEnd()
        glutSwapBuffers()

    def display(self) -> None:
        if self.closing or self.cuda_pbo is None:
            return
        try:
            self.update_latest_message()
            if self.pending_latest is not None:
                if self.copy_frame_to_pbo(self.pending_latest):
                    self.upload_pbo_to_texture()
            self.draw_fullscreen_quad()
            now = time.time()
            if now - self.last_status_time >= 1.0:
                latest = self.pending_latest.seq if self.pending_latest else None
                print(f"[viewer] latest={latest} shown={self.last_displayed_seq} displayed={self.displayed_count} dropped={self.dropped_count}", flush=True)
                self.last_status_time = now
        except Exception as e:
            print("[viewer] display error:", repr(e), flush=True)
            self.request_close()
            return
        if not self.closing:
            glutPostRedisplay()

    def request_close(self) -> None:
        self.closing = True
        self.cleanup()
        try:
            glutLeaveMainLoop()
        except Exception:
            os._exit(0)

    def keyboard(self, key, x, y) -> None:
        if key in (b"q", b"\x1b"):
            self.request_close()

    def cleanup(self) -> None:
        try:
            if self.cuda_pbo is not None:
                self.cuda_pbo.unregister(); self.cuda_pbo = None
        except Exception as e: print("[cleanup] cuda_pbo:", e, flush=True)
        try:
            if self.ipc_mem is not None:
                self.ipc_mem.close(); self.ipc_mem = None
        except Exception as e: print("[cleanup] ipc_mem:", e, flush=True)
        try:
            if self.pbo is not None:
                glDeleteBuffers(1, [self.pbo]); self.pbo = None
        except Exception as e: print("[cleanup] pbo:", e, flush=True)
        try:
            if self.tex is not None:
                glDeleteTextures([self.tex]); self.tex = None
        except Exception as e: print("[cleanup] tex:", e, flush=True)
        try:
            if self.cuda_ctx is not None:
                self.cuda_ctx.pop(); self.cuda_ctx.detach(); self.cuda_ctx = None
        except Exception as e: print("[cleanup] cuda_ctx:", e, flush=True)

    def run(self) -> None:
        try:
            self.attach_iox2_and_wait_info()
            self.init_gl_window()
            self.init_gl_resources()
            self.init_cuda_after_gl_context_exists()
            self.open_cuda_ipc_handle()
            glutDisplayFunc(self.display)
            glutKeyboardFunc(self.keyboard)
            try: glutCloseFunc(self.request_close)
            except Exception: pass
            print("[viewer] running. Press q or Esc to quit.", flush=True)
            glutMainLoop()
        except Exception:
            self.cleanup()
            raise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="iceoryx2 StreamInfo/FrameReady + CUDA IPC -> OpenGL viewer")
    p.add_argument("--stream", default="rgb")
    p.add_argument("--iox2-prefix", default="CudaIpcVideo")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--flip-y", action="store_true", default=True)
    p.add_argument("--no-flip-y", dest="flip_y", action="store_false")
    return p.parse_args()


def main() -> int:
    Iox2GlViewer(parse_args()).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

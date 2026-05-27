#!/usr/bin/env python3
import os
import sys
import time
import ctypes
import struct
from multiprocessing.shared_memory import SharedMemory

import numpy as np

# Required on Windows so pycuda._driver can find CUDA Toolkit DLLs.
CUDA_ROOT = os.environ.get("CUDA_PATH")
if CUDA_ROOT:
    os.add_dll_directory(os.path.join(CUDA_ROOT, "bin"))

from OpenGL.GL import *
from OpenGL.GLUT import *

import pycuda.driver as cuda
import pycuda.gl as cudagl
from pycuda.compiler import SourceModule


# Must match producer
SHM_NAME = "cuda_ipc_rgba_stream_demo_v1"
MAGIC = b"CIPCRGBA"

HEADER_FMT = "<8sIiiiiQQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

HANDLE_OFFSET = 256
CTRL_OFFSET = 512

LATEST_OFFSET = CTRL_OFFSET
STOP_OFFSET = CTRL_OFFSET + 8
GEN_OFFSET = CTRL_OFFSET + 16


# Runtime state
WIDTH = None
HEIGHT = None
NUM_SLOTS = None
CHANNELS = None
ITEMSIZE = None
FRAME_BYTES = None

shm = None
shm_buf = None

tex = None
pbo = None
cuda_pbo = None
cuda_ctx = None

ipc_mem = None
remote_base_ptr = None

copy_kernel = None

closing = False
last_displayed_seq = -1
last_status_time = 0.0
displayed_count = 0
dropped_count = 0


CUDA_SRC = r"""
extern "C" __global__
void copy_rgba_from_ipc_to_pbo(
    const unsigned char *src,
    unsigned char *dst,
    int width,
    int height,
    int flip_y
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int src_y = flip_y ? (height - 1 - y) : y;

    int src_i = 4 * (src_y * width + x);
    int dst_i = 4 * (y * width + x);

    dst[dst_i + 0] = src[src_i + 0];
    dst[dst_i + 1] = src[src_i + 1];
    dst[dst_i + 2] = src[src_i + 2];
    dst[dst_i + 3] = src[src_i + 3];
}
"""


def read_i64(buf, offset):
    return struct.unpack_from("<q", buf, offset)[0]


def gen_offset(slot):
    return GEN_OFFSET + slot * 8


def check_gl_error(where):
    err = glGetError()
    if err != GL_NO_ERROR:
        raise RuntimeError(f"OpenGL error at {where}: {err}")


def wait_for_shm_and_header(timeout_sec=30.0):
    global shm, shm_buf
    global WIDTH, HEIGHT, NUM_SLOTS, CHANNELS, ITEMSIZE, FRAME_BYTES

    deadline = time.time() + timeout_sec
    last_err = None

    while time.time() < deadline:
        try:
            shm = SharedMemory(name=SHM_NAME, create=False)
            shm_buf = shm.buf

            header = bytes(shm_buf[:HEADER_SIZE])
            magic, handle_len, width, height, num_slots, channels, itemsize, frame_bytes = struct.unpack(
                HEADER_FMT,
                header,
            )

            if magic != MAGIC:
                raise RuntimeError(f"bad shm magic: {magic!r}")

            if handle_len <= 0:
                raise RuntimeError("producer has not written CUDA IPC handle yet")

            if channels != 4 or itemsize != 1:
                raise RuntimeError(
                    f"this viewer expects RGBA8, got channels={channels}, itemsize={itemsize}"
                )

            WIDTH = int(width)
            HEIGHT = int(height)
            NUM_SLOTS = int(num_slots)
            CHANNELS = int(channels)
            ITEMSIZE = int(itemsize)
            FRAME_BYTES = int(frame_bytes)

            print("[viewer] attached shm:", SHM_NAME, flush=True)
            print("[viewer] frame:", WIDTH, "x", HEIGHT, "RGBA8", flush=True)
            print("[viewer] slots:", NUM_SLOTS, flush=True)
            print("[viewer] handle bytes:", handle_len, flush=True)
            return handle_len

        except FileNotFoundError as e:
            last_err = e
            time.sleep(0.2)
        except Exception as e:
            last_err = e
            time.sleep(0.2)

    raise RuntimeError(f"could not attach to producer shm: {last_err}")


def init_gl_window():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)
    glutInitWindowSize(WIDTH, HEIGHT)
    glutCreateWindow(b"CUDA IPC RGBA -> OpenGL PBO viewer")

    vendor = glGetString(GL_VENDOR)
    renderer = glGetString(GL_RENDERER)
    version = glGetString(GL_VERSION)

    print("[GL] vendor  :", vendor.decode(errors="replace") if vendor else None)
    print("[GL] renderer:", renderer.decode(errors="replace") if renderer else None)
    print("[GL] version :", version.decode(errors="replace") if version else None)

    glViewport(0, 0, WIDTH, HEIGHT)
    glDisable(GL_DEPTH_TEST)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    check_gl_error("init_gl_window")


def init_gl_resources():
    global tex, pbo

    # Texture displayed on screen.
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA8,
        WIDTH,
        HEIGHT,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        None,
    )

    # Pixel Buffer Object. CUDA writes into this.
    pbo = glGenBuffers(1)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo)
    glBufferData(
        GL_PIXEL_UNPACK_BUFFER,
        WIDTH * HEIGHT * 4,
        None,
        GL_STREAM_DRAW,
    )
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    check_gl_error("init_gl_resources")


def init_cuda_after_gl_context_exists():
    global cuda_ctx, copy_kernel

    cuda.init()
    dev = cuda.Device(0)

    # Requires the OpenGL context to already exist.
    cuda_ctx = cudagl.make_context(dev)

    mod = SourceModule(CUDA_SRC, options=["--use_fast_math"])
    copy_kernel = mod.get_function("copy_rgba_from_ipc_to_pbo")


def register_cuda_gl_resources():
    global cuda_pbo

    cuda_pbo = cudagl.RegisteredBuffer(
        int(pbo),
        cudagl.graphics_map_flags.WRITE_DISCARD,
    )


def open_cuda_ipc_handle(handle_len):
    global ipc_mem, remote_base_ptr

    handle_bytes = bytes(shm_buf[HANDLE_OFFSET: HANDLE_OFFSET + handle_len])

    print("[viewer] raw handle type:", type(handle_bytes), "len:", len(handle_bytes), flush=True)

    if len(handle_bytes) != 64:
        raise RuntimeError(f"expected 64-byte CUDA IPC handle, got {len(handle_bytes)}")

    # Important:
    # PyCUDA 2026.1 IPCMemoryHandle expects Python bytearray, not bytes.
    handle = bytearray(handle_bytes)

    ipc_mem = cuda.IPCMemoryHandle(handle)
    remote_base_ptr = int(ipc_mem)

    print("[viewer] opened IPC base ptr:", hex(remote_base_ptr), flush=True)


def copy_latest_ipc_frame_to_pbo():
    global last_displayed_seq, displayed_count, dropped_count

    latest = read_i64(shm_buf, LATEST_OFFSET)
    if latest < 0:
        return False

    if latest == last_displayed_seq:
        return False

    slot = latest % NUM_SLOTS
    expected_gen = latest * 2 + 2

    g1 = read_i64(shm_buf, gen_offset(slot))
    if g1 != expected_gen or (g1 & 1):
        dropped_count += 1
        return False

    src_ptr = remote_base_ptr + slot * FRAME_BYTES

    mapping = cuda_pbo.map()
    try:
        pbo_ptr, pbo_size = mapping.device_ptr_and_size()

        if pbo_size < FRAME_BYTES:
            raise RuntimeError(f"PBO too small: {pbo_size} < {FRAME_BYTES}")

        block = (16, 16, 1)
        grid = (
            (WIDTH + block[0] - 1) // block[0],
            (HEIGHT + block[1] - 1) // block[1],
            1,
        )

        copy_kernel(
            np.uintp(src_ptr),
            np.uintp(pbo_ptr),
            np.int32(WIDTH),
            np.int32(HEIGHT),
            np.int32(1),  # flip_y. Change to 0 if image appears upside down.
            block=block,
            grid=grid,
        )

        cuda.Context.synchronize()

    finally:
        mapping.unmap()

    g2 = read_i64(shm_buf, gen_offset(slot))
    if g1 != g2:
        # Producer overwrote while viewer copied. Do not upload this PBO.
        dropped_count += 1
        return False

    last_displayed_seq = latest
    displayed_count += 1
    return True


def upload_pbo_to_texture():
    glBindTexture(GL_TEXTURE_2D, tex)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo)

    glTexSubImage2D(
        GL_TEXTURE_2D,
        0,
        0,
        0,
        WIDTH,
        HEIGHT,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        ctypes.c_void_p(0),
    )

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)


def draw_fullscreen_quad():
    glClear(GL_COLOR_BUFFER_BIT)

    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex)

    glBegin(GL_QUADS)

    glTexCoord2f(0.0, 0.0)
    glVertex2f(-1.0, -1.0)

    glTexCoord2f(1.0, 0.0)
    glVertex2f(1.0, -1.0)

    glTexCoord2f(1.0, 1.0)
    glVertex2f(1.0, 1.0)

    glTexCoord2f(0.0, 1.0)
    glVertex2f(-1.0, 1.0)

    glEnd()
    glutSwapBuffers()


def display():
    global last_status_time

    if closing or cuda_pbo is None:
        return

    try:
        got_new = copy_latest_ipc_frame_to_pbo()
        if got_new:
            upload_pbo_to_texture()

        draw_fullscreen_quad()

        now = time.time()
        if now - last_status_time > 1.0:
            latest = read_i64(shm_buf, LATEST_OFFSET)
            stop = read_i64(shm_buf, STOP_OFFSET)
            print(
                f"[viewer] latest={latest} shown={last_displayed_seq} "
                f"displayed={displayed_count} dropped={dropped_count} stop={stop}",
                flush=True,
            )
            last_status_time = now

        if read_i64(shm_buf, STOP_OFFSET):
            print("[viewer] producer stop detected", flush=True)
            request_close()
            return

    except Exception as e:
        print("[viewer] display error:", repr(e), flush=True)
        request_close()
        return

    if not closing:
        glutPostRedisplay()


def request_close():
    global closing
    closing = True
    cleanup()

    try:
        glutLeaveMainLoop()
    except Exception:
        os._exit(0)


def keyboard(key, x, y):
    if key in (b"q", b"\x1b"):
        request_close()


def cleanup():
    global cuda_pbo, pbo, tex, cuda_ctx
    global ipc_mem, shm, shm_buf

    try:
        if cuda_pbo is not None:
            cuda_pbo.unregister()
            cuda_pbo = None
    except Exception as e:
        print("[cleanup] cuda_pbo:", e, flush=True)

    try:
        if ipc_mem is not None:
            ipc_mem.close()
            ipc_mem = None
    except Exception as e:
        print("[cleanup] ipc_mem:", e, flush=True)

    try:
        if pbo is not None:
            glDeleteBuffers(1, [pbo])
            pbo = None
    except Exception as e:
        print("[cleanup] pbo:", e, flush=True)

    try:
        if tex is not None:
            glDeleteTextures([tex])
            tex = None
    except Exception as e:
        print("[cleanup] tex:", e, flush=True)

    try:
        if cuda_ctx is not None:
            cuda_ctx.pop()
            cuda_ctx.detach()
            cuda_ctx = None
    except Exception as e:
        print("[cleanup] cuda_ctx:", e, flush=True)

    try:
        if shm_buf is not None:
            shm_buf.release()
            shm_buf = None
    except Exception:
        pass

    try:
        if shm is not None:
            shm.close()
            shm = None
    except Exception as e:
        print("[cleanup] shm:", e, flush=True)


def main():
    try:
        handle_len = wait_for_shm_and_header()

        init_gl_window()
        init_gl_resources()

        init_cuda_after_gl_context_exists()
        register_cuda_gl_resources()
        open_cuda_ipc_handle(handle_len)

        glutDisplayFunc(display)
        glutKeyboardFunc(keyboard)

        try:
            glutCloseFunc(request_close)
        except Exception:
            pass

        print("[viewer] running. Press q or Esc to quit.", flush=True)
        glutMainLoop()

    except Exception:
        cleanup()
        raise


if __name__ == "__main__":
    main()
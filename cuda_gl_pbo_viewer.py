#!/usr/bin/env python3
import os; os.add_dll_directory(os.environ['CUDA_PATH'] + r'\bin');
import sys
import ctypes
import numpy as np

from OpenGL.GL import *
from OpenGL.GLUT import *

import pycuda.driver as cuda
import pycuda.gl as cudagl
from pycuda.compiler import SourceModule


WIDTH = 640
HEIGHT = 480

tex = None
pbo = None
cuda_pbo = None
cuda_ctx = None
fill_kernel = None
frame = 0


CUDA_SRC = r"""
extern "C" __global__
void fill_rgba(unsigned char *pbo, int width, int height, int frame)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int i = 4 * (y * width + x);

    unsigned char r = (unsigned char)((x + frame) & 255);
    unsigned char g = (unsigned char)((y + frame) & 255);
    unsigned char b = (unsigned char)(((x ^ y) + frame) & 255);

    pbo[i + 0] = r;
    pbo[i + 1] = g;
    pbo[i + 2] = b;
    pbo[i + 3] = 255;
}
"""


def check_gl_error(where):
    err = glGetError()
    if err != GL_NO_ERROR:
        raise RuntimeError(f"OpenGL error at {where}: {err}")


def init_gl():
    global tex, pbo, cuda_pbo

    glViewport(0, 0, WIDTH, HEIGHT)
    glDisable(GL_DEPTH_TEST)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    # Texture that will be displayed.
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

    # Pixel Buffer Object. CUDA will write here.
    pbo = glGenBuffers(1)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo)
    glBufferData(
        GL_PIXEL_UNPACK_BUFFER,
        WIDTH * HEIGHT * 4,
        None,
        GL_STREAM_DRAW,
    )
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    check_gl_error("init_gl")
    
    vendor = glGetString(GL_VENDOR)
    renderer = glGetString(GL_RENDERER)
    version = glGetString(GL_VERSION)

    print("[GL] vendor  :", vendor.decode(errors="replace") if vendor else None)
    print("[GL] renderer:", renderer.decode(errors="replace") if renderer else None)
    print("[GL] version :", version.decode(errors="replace") if version else None)

def register_cuda_gl_resources():
    global cuda_pbo

    cuda_pbo = cudagl.RegisteredBuffer(
        int(pbo),
        cudagl.graphics_map_flags.WRITE_DISCARD,
    )

def init_cuda_after_gl_context_exists():
    global cuda_ctx, fill_kernel

    # Important: OpenGL context already exists before this.
    cuda.init()
    dev = cuda.Device(0)

    # Create CUDA context with GL interoperability enabled.
    cuda_ctx = cudagl.make_context(dev)

    mod = SourceModule(CUDA_SRC, options=["--use_fast_math"])
    fill_kernel = mod.get_function("fill_rgba")


def cuda_fill_pbo():
    global frame

    # Map GL PBO into CUDA address space.
    mapping = cuda_pbo.map()
    try:
        dev_ptr, size = mapping.device_ptr_and_size()
        assert size >= WIDTH * HEIGHT * 4

        block = (16, 16, 1)
        grid = (
            (WIDTH + block[0] - 1) // block[0],
            (HEIGHT + block[1] - 1) // block[1],
            1,
        )

        fill_kernel(
            np.uintp(dev_ptr),
            np.int32(WIDTH),
            np.int32(HEIGHT),
            np.int32(frame),
            block=block,
            grid=grid,
        )

        cuda.Context.synchronize()
        frame += 1

    finally:
        # OpenGL must not use the PBO while CUDA has it mapped.
        mapping.unmap()


def upload_pbo_to_texture():
    glBindTexture(GL_TEXTURE_2D, tex)

    # With GL_PIXEL_UNPACK_BUFFER bound, the final argument is an offset
    # into the PBO, not a CPU pointer.
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

closing = False
def display():
    if closing or cuda_pbo is None:
        return

    cuda_fill_pbo()
    upload_pbo_to_texture()
    draw_fullscreen_quad()

    if not closing:
        glutPostRedisplay()


def keyboard(key, x, y):
    global closing

    if key in (b"q", b"\x1b"):
        closing = True
        cleanup()

        # freeglut supports this. PyOpenGL may expose it.
        try:
            glutLeaveMainLoop()
        except Exception:
            # Last-resort clean exit for GLUT on Windows.
            import os
            os._exit(0)


def cleanup():
    global cuda_pbo, pbo, tex, cuda_ctx

    try:
        if cuda_pbo is not None:
            cuda_pbo.unregister()
            cuda_pbo = None
    except Exception:
        pass

    try:
        if pbo is not None:
            glDeleteBuffers(1, [pbo])
            pbo = None
    except Exception:
        pass

    try:
        if tex is not None:
            glDeleteTextures([tex])
            tex = None
    except Exception:
        pass

    try:
        if cuda_ctx is not None:
            cuda_ctx.pop()
            cuda_ctx.detach()
            cuda_ctx = None
    except Exception:
        pass


def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)
    glutInitWindowSize(WIDTH, HEIGHT)
    glutCreateWindow(b"CUDA -> OpenGL PBO viewer")

    init_gl()

    # Must happen after glutCreateWindow.
    init_cuda_after_gl_context_exists()

    # Must happen after CUDA/GL context exists.
    register_cuda_gl_resources()

    glutDisplayFunc(display)
    glutKeyboardFunc(keyboard)
    glutCloseFunc(cleanup)

    print("Running. Press q or Esc to quit.")
    glutMainLoop()

if __name__ == "__main__":
    main()
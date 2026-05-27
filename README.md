# CUDA IPC GPU Video Experiments

Experiments for moving frames between Python processes without copying video
pixels through host memory. The data path stays on the NVIDIA GPU through CUDA
IPC; small control messages such as allocation handles, frame sequence numbers,
and slot state travel through shared memory or iceoryx2.

The repository progresses from small CUDA IPC examples to a multi-consumer
video pipeline:

```text
TCP video source / camera
    -> PyNvVideoCodec hardware decode
    -> CUDA-resident CuPy ring buffer
    -> CUDA IPC handle exported once
    -> OpenGL display and/or Torch inference consumer
```

## What Is Here

| Path | Purpose |
| --- | --- |
| `cupy_cuda_ipc_shm_demo.py` | Minimal two-process CuPy CUDA IPC example: one process exports memory and the other modifies it. |
| `cupy_cuda_ipc_stream_readonly_demo.py` | One producer and one read-only consumer using a small GPU ring buffer. |
| `cupy_cuda_ipc_stream_multi_consumers.py` | Extends the ring example to several independent consumers. |
| `cuda_gl_pbo_viewer.py` | CUDA/OpenGL interoperability smoke test that fills an OpenGL PBO from a CUDA kernel. |
| `IPCrgba/` | Synthetic RGBA CUDA IPC producer and a PyCUDA/OpenGL viewer. |
| `TCPsteam/` | Hardware-decoded TCP video published through CUDA IPC, with Python shared memory as the control plane. |
| `TCPiceoryx2/` | The same GPU data-plane idea with iceoryx2 pub/sub for metadata and frame notifications. |
| `external/pycuda/` | Vendored PyCUDA source used for a build with CUDA/OpenGL interoperability enabled. |

## Design

CUDA IPC exports a GPU allocation from the producer and lets consumers map that
allocation in their own CUDA contexts. In the stream demos, the allocation is a
ring of frame slots. The producer publishes only metadata and synchronization
state on the CPU side:

```text
producer:
  mark slot generation odd       # write in progress
  copy/write frame on GPU
  synchronize CUDA work
  mark slot generation even      # frame is stable
  publish latest sequence number

consumer:
  select a published slot
  verify the generation
  display or clone/process on GPU
  verify the generation again
```

This is a real-time, latest-frame-oriented design. A consumer that falls behind
may drop frames after its slot has been reused. It is not a lossless queue.

The original producer allocation must stay alive until every consumer closes
its imported CUDA IPC mapping. The examples deliberately keep ownership in the
producer and coordinate shutdown around that requirement.

## Requirements

The current demo workflow is oriented toward Windows and an NVIDIA CUDA/OpenGL
desktop environment.

- NVIDIA GPU and CUDA Toolkit, with `CUDA_PATH` set.
- Python `3.13` and [uv](https://docs.astral.sh/uv/).
- FFmpeg available on `PATH` for the simulated encoded TCP source.
- An OpenGL/GLUT runtime for viewer windows.
- PyCUDA built with CUDA/OpenGL interop support for the viewer scripts.
- `PyNvVideoCodec` for hardware video decode.
- `iceoryx2` for the iceoryx2 control-plane variant.

The Python dependencies tracked in `pyproject.toml` include CuPy for CUDA 12,
CUDA-enabled Torch, PyNvVideoCodec, OpenGL bindings, and iceoryx2:

```bat
uv sync
```

The OpenGL viewers additionally require an importable PyCUDA build with GL
interop enabled (`--cuda-enable-gl`). Source and generic build instructions are
included under `external\pycuda`, but the CUDA/compiler paths are specific to
your machine. Run viewer commands from an **x64 Native Tools Command Prompt for
VS 2022** so NVCC can find the Microsoft compiler.

## Start Small

These scripts establish the CUDA IPC behavior without requiring decoded video.

### Memory sharing

One child process creates a GPU array and exports it; a second process maps it
and writes back into the same allocation:

```bat
uv run cupy_cuda_ipc_shm_demo.py
```

### Ring-buffer streaming

Run a read-only consumer against a rotating CUDA allocation:

```bat
uv run cupy_cuda_ipc_stream_readonly_demo.py
```

Try the same publication scheme with three independent consumers:

```bat
uv run cupy_cuda_ipc_stream_multi_consumers.py
```

### CUDA/OpenGL smoke test

This verifies CUDA writing directly to an OpenGL Pixel Buffer Object before
introducing inter-process memory:

```bat
uv run cuda_gl_pbo_viewer.py
```

Press `q` or `Esc` to close the viewer.

## Synthetic RGBA Viewer

`IPCrgba` creates a continuously changing `640 x 480` RGBA8 frame ring in
CuPy. The viewer maps the remote CUDA allocation using PyCUDA and copies the
latest frame into an OpenGL PBO on the GPU.

Open two x64 Native Tools terminals from the repository root:

```bat
uv run IPCrgba\ipc_rgba_producer_cupy.py
```

```bat
uv run IPCrgba\ipc_rgba_gl_viewer_pycuda.py
```

The processes coordinate through shared memory named
`cuda_ipc_rgba_stream_demo_v1`. Stop the producer with `Ctrl+C`; close the
viewer with `q` or `Esc`.

## TCP Video With Shared-Memory Control

`TCPsteam` simulates encoded camera streams, hardware-decodes them to GPU
frames, copies them into persistent CUDA IPC rings, and permits display or AI
consumption without a CPU pixel copy.

```text
TCP simulator -> PyNvVideoCodec -> CuPy CUDA IPC ring
                                -> PyCUDA/OpenGL viewer
                                -> CuPy/Torch AI consumer
```

Open separate x64 Native Tools terminals from the repository root:

```bat
uv run TCPsteam\tools\tcp_sim_src.py --streams rgb
```

```bat
uv run TCPsteam\producers\video_decode_ipc_publisher.py --ip 127.0.0.1 --streams rgb
```

```bat
uv run TCPsteam\consumers\ipc_gl_viewer_pycuda.py --stream rgb
```

Optional Torch consumer:

```bat
uv run TCPsteam\consumers\ipc_ai_consumer_torch.py --stream rgb --latest-only
```

For the simulator, `rgb` is HEVC on port `5000`; `left` and `right` are H.264
on ports `5001` and `5002`. To publish all three:

```bat
uv run TCPsteam\tools\tcp_sim_src.py --streams rgb,left,right
uv run TCPsteam\producers\video_decode_ipc_publisher.py --ip 127.0.0.1 --streams rgb,left,right
```

Each stream obtains its own shared-memory control object:

```text
cuda_ipc_stream_rgb_v1
cuda_ipc_stream_left_v1
cuda_ipc_stream_right_v1
```

Decoded ring slots are unsigned-byte CHW frames: `RGBP_CHW`, `GRAY_CHW`, or
`RGBA_CHW`. The OpenGL consumer performs conversion to RGBA8 in a CUDA kernel.
The Torch consumer clones the selected slot on the GPU before inference so the
producer can reuse ring slots without mutating in-flight model input.

## TCP Video With iceoryx2 Control

`TCPiceoryx2` keeps the same CUDA IPC data plane, but replaces Python shared
memory notifications with iceoryx2 services:

```text
CUDA IPC = GPU frame storage
iceoryx2 = stream information and frame-ready notifications
```

Run an RGB stream in separate terminals:

```bat
uv run TCPiceoryx2\tools\tcp_sim_src.py --streams rgb
```

```bat
uv run TCPiceoryx2\producers\video_decode_iox2_publisher.py --ip 127.0.0.1 --streams rgb --num-slots 16
```

```bat
uv run TCPiceoryx2\consumers\iox2_gl_viewer_pycuda.py --stream rgb
```

Optional Torch consumer:

```bat
uv run TCPiceoryx2\consumers\iox2_ai_consumer_torch.py --stream rgb --latest-only
```

For stream `rgb`, the default iceoryx2 services are:

```text
CudaIpcVideo/rgb/StreamInfo
CudaIpcVideo/rgb/FrameReady
```

Use `--iox2-prefix` on both producer and consumers to select another service
namespace. `--num-slots` trades GPU memory for tolerance of slower consumers;
the iceoryx2 publisher defaults to `16` slots.

## Extending The Pipeline

- Replace `DummyAI` in the Torch consumer scripts with the intended model.
- Use `--video-file` on a TCP simulator to stream a local OpenCV-readable
  recording instead of its generated frames.
- Use `--streams rgb,left,right` for three concurrent encoded sources.
- Add consumer acknowledgements before slot reuse if every frame must be
  processed. The shipped generation checks detect overwrite but intentionally
  do not prevent it.

## Notes And Limitations

- CUDA IPC mappings are device-specific; producer and consumers should select
  the same GPU with their `--gpu-id` arguments.
- The viewers need a desktop OpenGL context and do not suit a headless session
  without additional display setup.
- CUDA IPC shares allocations, not ownership or synchronization. Do not let the
  producer exit while consumers still access its ring.
- The project is a set of executable experiments rather than a packaged
  library API. The scripts are intended to make data flow and synchronization
  choices explicit and easy to modify.

Subdirectory notes remain available in `TCPsteam\README.md` and
`TCPiceoryx2\README_iceoryx2.md` for focused reference.

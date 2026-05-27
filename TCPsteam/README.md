# CUDA IPC Video Stream Demo

This repo merges three pieces:

1. `tools/tcp_sim_src.py`  
   OpenCV/ffmpeg simulator that emits encoded TCP streams:
   - RGB HEVC on port 5000
   - left H.264 on port 5001
   - right H.264 on port 5002

2. `producers/video_decode_ipc_publisher.py`  
   Reads encoded TCP stream, decodes on NVIDIA GPU with `PyNvVideoCodec`, imports the decoded frame to Torch with DLPack, copies it into a persistent CuPy CUDA IPC ring buffer, and publishes the ring metadata/handle via Python shared memory.

3. Consumers:
   - `consumers/ipc_gl_viewer_pycuda.py`: CUDA IPC -> CUDA/OpenGL PBO -> display
   - `consumers/ipc_ai_consumer_torch.py`: CUDA IPC -> CuPy -> Torch -> DummyAI
   - `consumers/ipc_debug_consumer_cupy.py`: small scalar-stats debug consumer

## Main pipeline

```text
tcp_sim_src.py / camera
    -> encoded TCP H.265/H.264
    -> PyNvVideoCodec GPU decode
    -> torch.from_dlpack(frame)
    -> persistent CuPy CUDA IPC ring buffer
    -> OpenGL viewer and/or Torch AI consumer
```

The publisher does **not** export decoder-owned frame memory directly. It owns a persistent ring buffer, copies decoded frames into it on GPU, then exports that ring.

## Requirements

Python packages you likely need:

```bat
uv add cupy-cuda12x torch pynvvideocodec PyOpenGL PyOpenGL_accelerate opencv-python numpy
```

For the OpenGL viewer with PyCUDA GL interop, you already built PyCUDA with GL enabled. Run viewer commands from an `x64 Native Tools Command Prompt for VS 2022` so NVCC can find `cl.exe`.

On Windows, make sure `CUDA_PATH` is set. The viewer adds `%CUDA_PATH%\bin` with `os.add_dll_directory()`.

## Run RGB demo

Open separate terminals.

### 1. Start simulator

```bat
uv run python tools\tcp_sim_src.py --streams rgb
```

### 2. Start GPU decode + IPC publisher

```bat
uv run python producers\video_decode_ipc_publisher.py --ip 127.0.0.1 --streams rgb
```

This creates shared memory named:

```text
cuda_ipc_stream_rgb_v1
```

### 3. Start OpenGL viewer

```bat
uv run python consumers\ipc_gl_viewer_pycuda.py --stream rgb
```

### 4. Optional Torch AI consumer

```bat
uv run python consumers\ipc_ai_consumer_torch.py --stream rgb --latest-only
```

## Run all streams

```bat
uv run python tools\tcp_sim_src.py --streams rgb,left,right
uv run python producers\video_decode_ipc_publisher.py --ip 127.0.0.1 --streams rgb,left,right
uv run python consumers\ipc_gl_viewer_pycuda.py --stream rgb
uv run python consumers\ipc_gl_viewer_pycuda.py --stream left
uv run python consumers\ipc_gl_viewer_pycuda.py --stream right
```

Each stream gets its own shared memory object:

```text
cuda_ipc_stream_rgb_v1
cuda_ipc_stream_left_v1
cuda_ipc_stream_right_v1
```

## Shared format

Current shared GPU frame format is CHW uint8:

```text
RGBP_CHW:  [slot, 3, H, W]
GRAY_CHW:  [slot, 1, H, W]
RGBA_CHW:  [slot, 4, H, W]
```

The OpenGL viewer converts CHW uint8 to RGBA8 PBO in a CUDA kernel.

## Synchronization model

This is a best-effort/latest-frame broadcast ring:

```text
producer:
  gen[slot] = seq*2+1       # writing
  GPU copy into slot
  cuda synchronize
  gen[slot] = seq*2+2       # stable
  latest_seq = seq

consumer:
  latest = latest_seq
  slot = latest % num_slots
  check generation before reading
  copy/process on GPU
  check generation after reading
```

If the producer is faster than a consumer, frames can be dropped. This is usually right for live video.

For strict "every consumer must see every frame", add per-consumer acknowledgements and let the producer reuse a slot only after all required consumers ack it.

## Where to plug in real AI

Replace `DummyAI` in `consumers/ipc_ai_consumer_torch.py` with your model.

The consumer clones the frame on GPU before inference:

```python
t_view = torch.utils.dlpack.from_dlpack(cp_frame)
t_copy = t_view.clone()
```

That clone prevents producer slot reuse from corrupting model input while inference is running.

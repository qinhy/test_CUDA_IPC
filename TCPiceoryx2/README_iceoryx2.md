# iceoryx2 Control Plane for CUDA IPC Video

This version keeps video pixels in GPU memory and uses iceoryx2 only for metadata/pub-sub.

```text
TCP camera / simulator
  -> PyNvVideoCodec GPU decode
  -> persistent CuPy CUDA IPC ring
  -> iceoryx2 StreamInfo message: CUDA IPC handle + shape/layout
  -> iceoryx2 FrameReady messages: seq/slot/timestamp
  -> OpenGL viewer / AI consumers open CUDA IPC handle and read GPU slots
```

## Why not publish pixels through iceoryx2?

iceoryx2 shared memory is host memory. Publishing video pixels through it would create:

```text
GPU -> CPU shared memory -> GPU/display/model
```

This repo keeps:

```text
CUDA IPC = GPU data plane
iceoryx2 = pub/sub control plane
```

## Install

```bat
uv add iceoryx2
```

Keep your manually built `pycuda` with `--cuda-enable-gl` for the OpenGL viewer.

## Run RGB demo

Open separate terminals.

### 1. Simulator

```bat
uv run TCPiceoryx2\tools\tcp_sim_src.py --streams rgb
```

### 2. Decode + CUDA IPC ring + iceoryx2 metadata

```bat
uv run TCPiceoryx2\producers\video_decode_iox2_publisher.py --ip 127.0.0.1 --streams rgb --num-slots 16
```

### 3. OpenGL viewer

```bat
uv run TCPiceoryx2\consumers\iox2_gl_viewer_pycuda.py --stream rgb
```

### 4. Optional Torch AI consumer

```bat
uv run TCPiceoryx2\consumers\iox2_ai_consumer_torch.py --stream rgb --latest-only
```

## iceoryx2 services

For stream `rgb`, default services are:

```text
CudaIpcVideo/rgb/StreamInfo
CudaIpcVideo/rgb/FrameReady
```

Change namespace with `--iox2-prefix`.

## Semantics

This version is latest-frame best effort. A slow consumer may miss frames or, in the worst case, read a slot after it has been reused. The default `--num-slots 16` reduces that risk for real-time preview/AI.

For strict no-overwrite semantics, add a consumer ACK service and make the producer reuse a slot only after all required consumers acknowledge the frame sequence.

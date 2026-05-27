from __future__ import annotations

import torch


def decoded_tensor_to_chw_uint8(t: torch.Tensor) -> torch.Tensor:
    """
    Normalize a decoded PyNvVideoCodec tensor to CUDA uint8 CHW.

    Accepts:
      [C,H,W], [H,W,C], or [H,W]
    Returns:
      contiguous uint8 [C,H,W] on CUDA, with C in {1,3,4}.
    """
    if not t.is_cuda:
        raise RuntimeError(f"expected CUDA tensor, got device={t.device}")

    if t.dtype != torch.uint8:
        t = t.to(dtype=torch.uint8)

    if t.ndim == 2:
        x = t.unsqueeze(0)

    elif t.ndim == 3:
        # Planar RGBP/CHW from PyNvVideoCodec is the intended fast path.
        if t.shape[0] in (1, 3, 4):
            x = t
        elif t.shape[-1] in (1, 3, 4):
            x = t.permute(2, 0, 1)
        else:
            raise RuntimeError(f"cannot infer channel dimension from shape={tuple(t.shape)}")
    else:
        raise RuntimeError(f"unexpected decoded tensor shape={tuple(t.shape)}")

    if x.shape[0] not in (1, 3, 4):
        raise RuntimeError(f"expected 1, 3, or 4 channels, got shape={tuple(x.shape)}")

    return x.contiguous()


def torch_tensor_to_cupy_view(t: torch.Tensor):
    """
    Return a CuPy view of a CUDA torch tensor using DLPack.
    The returned CuPy array aliases the torch tensor memory.
    """
    import cupy as cp

    if not t.is_cuda:
        raise RuntimeError("torch tensor must be CUDA")

    # Newer CuPy supports Python DLPack protocol directly.
    try:
        return cp.from_dlpack(t)
    except Exception:
        # Compatibility fallback.
        return cp.fromDlpack(torch.utils.dlpack.to_dlpack(t))


def preprocess_for_ai(frame_chw: torch.Tensor) -> torch.Tensor:
    """
    frame_chw: uint8 CUDA tensor [C,H,W], C=1/3/4
    returns: float32 CUDA tensor [1,3,H,W] in 0..1
    """
    if frame_chw.ndim != 3:
        raise RuntimeError(f"expected [C,H,W], got {tuple(frame_chw.shape)}")

    x = frame_chw

    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    elif x.shape[0] == 4:
        x = x[:3]
    elif x.shape[0] != 3:
        raise RuntimeError(f"expected 1, 3, or 4 channels, got {x.shape[0]}")

    return x.unsqueeze(0).float().div_(255.0)

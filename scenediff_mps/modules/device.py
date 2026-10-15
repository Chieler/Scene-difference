"""MPS port: device helpers replacing hardcoded torch.cuda calls."""

import torch


def empty_device_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def autocast_dtype(device):
    """bf16 where the original used it (Ampere+ CUDA); MPS supports bf16 autocast too."""
    device = torch.device(device)
    if device.type == 'cuda':
        return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    return torch.bfloat16


def scatter_mean(src, index, dim=0):
    """Pure-torch replacement for torch_scatter.scatter_mean along dim 0 (empty bins -> 0)."""
    assert dim == 0
    n = int(index.max()) + 1 if index.numel() else 0
    out = torch.zeros((n,) + tuple(src.shape[1:]), dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    count = torch.zeros(n, dtype=src.dtype, device=src.device)
    count.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return out / count.clamp(min=1).view(-1, *([1] * (src.dim() - 1)))

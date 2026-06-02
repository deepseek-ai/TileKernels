"""Python wrappers for CUDA MHC kernels (inference only)."""

import sys
import os
import torch


def _get_extension():
    _dir = os.path.dirname(os.path.abspath(__file__))
    _parent = os.path.dirname(_dir)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    import _mhc_cuda
    return _mhc_cuda


def mhc_expand_cuda(input: torch.Tensor) -> torch.Tensor:
    M, H = input.shape
    output = torch.empty(M, 4, H, dtype=input.dtype, device=input.device)
    _get_extension().mhc_expand_cuda(input, output)
    return output


def mhc_pre_fused_cuda(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    rms_eps: float = 1e-6,
    pre_eps: float = 1e-6,
    sinkhorn_eps: float = 1e-6,
    post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 10,
) -> tuple:
    layer_input, post_mix, comb_mix = _get_extension().mhc_pre_fused_cuda(
        residual, fn, scale, base,
        rms_eps, pre_eps, sinkhorn_eps, post_mult_value, sinkhorn_repeat)
    return layer_input, post_mix, comb_mix


def mhc_post_cuda(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
) -> torch.Tensor:
    return _get_extension().mhc_post_cuda(x, residual, post_mix, comb_mix)


def mhc_head_cuda(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    rms_eps: float = 1e-6,
    pre_eps: float = 1e-6,
) -> torch.Tensor:
    return _get_extension().mhc_head_cuda(residual, fn, scale, base, rms_eps, pre_eps)

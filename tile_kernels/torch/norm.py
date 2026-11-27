"""RMSNorm mathematical references.

Scalar operation grouping follows CUDA. PyTorch reductions and unfused
operations do not reproduce the device kernels bitwise; compare with a
numerical tolerance.
"""

from typing import Optional

import torch


def norm_forward_ref(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    residual: Optional[torch.Tensor] = None,
    out_scale: float = 1.0,
    out_dtype: Optional[torch.dtype] = None,
    rms_group_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_dtype = x.dtype if out_dtype is None else out_dtype
    residual_out = x if residual is None else (x + residual).to(x.dtype)
    rms_group_size = x.shape[-1] if rms_group_size is None else rms_group_size
    norm_input = residual_out.float().reshape(-1, rms_group_size)
    rstd = (norm_input.square().sum(dim=-1) / rms_group_size + eps).rsqrt()
    normalized = norm_input * rstd[..., None]
    out = (normalized.reshape_as(x) * ((weight.float() if weight is not None else 1.0) * out_scale)).to(out_dtype)
    return out, rstd, residual_out


def norm_backward_ref(
    out_grad: torch.Tensor,
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    rstd: torch.Tensor,
    residual_out_grad: Optional[torch.Tensor] = None,
    out_scale: float = 1.0,
    x_grad_dtype: Optional[torch.dtype] = None,
    rms_group_size: Optional[int] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    x_grad_dtype = x.dtype if x_grad_dtype is None else x_grad_dtype
    rms_group_size = x.shape[-1] if rms_group_size is None else rms_group_size
    num_tokens, hidden = x.shape
    num_groups = hidden // rms_group_size
    rstd = rstd.reshape(num_tokens, num_groups, 1)
    xhat = x.float().reshape(num_tokens, num_groups, rms_group_size) * rstd
    scaled_out_grad = out_grad.float().reshape_as(xhat) * out_scale
    weighted_out_grad = scaled_out_grad * (weight.float().reshape(num_groups, rms_group_size) if weight is not None else 1.0)
    xhat_scaled_out_grad = xhat * scaled_out_grad
    mean_xhat_weighted_out_grad = (xhat * weighted_out_grad).sum(dim=-1, keepdim=True) / rms_group_size
    x_grad = ((weighted_out_grad - xhat * mean_xhat_weighted_out_grad) * rstd).reshape_as(x)
    if residual_out_grad is not None:
        x_grad = x_grad + residual_out_grad.float()
    weight_grad = xhat_scaled_out_grad.sum(dim=0).reshape(hidden) if weight is not None else None
    return x_grad.to(x_grad_dtype), weight_grad

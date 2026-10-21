from typing import Optional

import torch
from tilelang import language as T

from tile_kernels.config import get_num_sms, get_num_vec_cores, is_ascend
from tile_kernels.quant.norm_backward_cuda import get_norm_backward_kernel_cuda


def norm_backward(
    out_grad: torch.Tensor,
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    rstd: torch.Tensor,
    residual_out_grad: Optional[torch.Tensor] = None,
    out_scale: float = 1.0,
    weight_grad: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute RMSNorm gradients from inputs saved by norm_forward.

    Args:
        out_grad: Output gradient, shape (num_tokens, hidden_size).
        x: Pre-normalization input returned by norm_forward.
        weight: Optional normalization weight; None means unit weight.
        rstd: Reciprocal RMS returned by norm_forward.
        residual_out_grad: Optional gradient of the pre-normalization input.
        out_scale: Forward output multiplier.
        weight_grad: Optional buffer to accumulate the weight gradient into.

    Returns:
        (x_grad, weight_grad). A supplied weight_grad is returned in place.
        When forward adds a residual, its gradient is also x_grad.
    """
    assert x.ndim == 2 and x.dtype in (torch.bfloat16, torch.float32) and x.stride(-1) == 1
    num_tokens, hidden_size = x.shape
    if is_ascend():
        assert 0 < hidden_size <= 7168 and hidden_size % 128 == 0, 'hidden_size must be a multiple of 128 up to 7168 on Ascend'
    else:
        assert 0 < hidden_size <= 8192 and hidden_size % 32 == 0, 'hidden_size must be a multiple of 32 in [32, 8192] on CUDA'
    assert out_grad.shape == x.shape and out_grad.dtype == x.dtype and out_grad.device == x.device and out_grad.stride(-1) == 1
    assert rstd.shape == (num_tokens,) and rstd.dtype == torch.float32 and rstd.device == x.device and rstd.is_contiguous()
    if residual_out_grad is not None:
        assert residual_out_grad.shape == x.shape and residual_out_grad.dtype == x.dtype and residual_out_grad.device == x.device
        assert residual_out_grad.stride(-1) == 1
    if weight is not None:
        assert weight.shape == (hidden_size,) and weight.dtype == x.dtype and weight.device == x.device and weight.is_contiguous()
    if weight_grad is not None:
        assert weight is not None
        assert weight_grad.shape == weight.shape and weight_grad.dtype in (torch.bfloat16, torch.float32) and weight_grad.device == x.device
        assert weight_grad.is_contiguous()

    x_grad = torch.empty_like(x, memory_format=torch.contiguous_format)
    weight_grad_initialized = weight_grad is not None
    if weight is not None and weight_grad is None:
        weight_grad = torch.empty_like(weight)
    if num_tokens == 0:
        if weight is not None and not weight_grad_initialized:
            weight_grad.zero_()
        return x_grad, weight_grad

    num_partials = get_num_vec_cores() if is_ascend() else get_num_sms()
    weight_grad_partial = None if weight is None else torch.empty((num_partials, hidden_size), dtype=torch.float32, device=x.device)
    kernel_kwargs = dict(
        hidden=hidden_size,
        with_weight=weight is not None,
        with_residual_out_grad=residual_out_grad is not None,
        weight_grad_initialized=weight_grad_initialized,
        out_scale=out_scale,
        x_dtype=T.dtype(x.dtype),
        weight_grad_dtype=T.dtype(x.dtype if weight_grad is None else weight_grad.dtype),
    )
    if is_ascend():
        from tile_kernels.quant.norm_backward_asc import get_norm_backward_kernel_asc

        kernel = get_norm_backward_kernel_asc(num_vec_cores=num_partials, **kernel_kwargs)
        args = (x, weight, out_grad, rstd, x_grad, weight_grad_partial, residual_out_grad, weight_grad)
    else:
        for tensor in (x, out_grad, residual_out_grad):
            if tensor is not None:
                assert tensor.stride(0) % 16 == 0
        # Grid sync uses a cooperative launch, which TileLang cannot combine with its PDL launch path.
        kernel = get_norm_backward_kernel_cuda(**kernel_kwargs, num_sms=num_partials)
        args = (x, weight, out_grad, rstd, x_grad, weight_grad_partial, weight_grad, residual_out_grad)

    kernel(*args)

    return x_grad, weight_grad

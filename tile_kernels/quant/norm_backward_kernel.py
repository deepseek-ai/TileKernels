from typing import Optional

import torch
from tilelang import language as T

from tile_kernels.config import get_num_sms, get_num_vec_cores, is_ascend
from tile_kernels.quant.norm_backward_cuda import _get_norm_backward_schedule, get_norm_backward_kernel_cuda


def norm_backward(
    out_grad: torch.Tensor,
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    rstd: torch.Tensor,
    residual_out_grad: Optional[torch.Tensor] = None,
    out_scale: float = 1.0,
    weight_grad: Optional[torch.Tensor] = None,
    x_grad_dtype: Optional[torch.dtype] = None,
    rms_group_size: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    with_weight_grad: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute RMSNorm gradients from inputs saved by norm_forward.

    Args:
        out_grad: BF16/FP32 output gradient, shape (num_tokens, hidden).
        x: Pre-normalization input returned by norm_forward.
        weight: Optional normalization weight; None means unit weight.
        rstd: Flat reciprocal RMS returned by norm_forward, with x.numel() // rms_group_size elements.
        residual_out_grad: Optional gradient of the pre-normalization input, matching x.dtype.
        out_scale: Forward output multiplier.
        weight_grad: Optional buffer to accumulate the weight gradient into.
        x_grad_dtype: Optional dtype for x_grad; None means ``x.dtype``.
        rms_group_size: RMS group width; defaults to hidden. Must divide hidden and be a multiple of 32 on CUDA or 128 on Ascend.
        out: Optional contiguous x.dtype buffer for recomputing the normalized forward output.
        with_weight_grad: Compute gamma gradients (default True). Callers may disable this for frozen weights.

    Returns:
        (x_grad, weight_grad). A supplied weight_grad is returned in place.
        When forward adds a residual, its gradient is also x_grad.
    """
    assert x.ndim == 2 and x.dtype in (torch.bfloat16, torch.float32) and x.stride(-1) == 1
    num_tokens, hidden = x.shape
    rms_group_size = hidden if rms_group_size is None else rms_group_size
    assert rms_group_size > 0 and hidden % rms_group_size == 0 and rms_group_size % 32 == 0
    num_rms_groups = hidden // rms_group_size
    with_weight_grad = with_weight_grad and weight is not None
    if is_ascend():
        assert 0 < hidden and hidden % 128 == 0, 'hidden must be a positive multiple of 128 on Ascend'
        assert rms_group_size % 128 == 0, 'rms_group_size must be a multiple of 128 on Ascend'
    else:
        assert 0 < hidden and hidden % 32 == 0, 'hidden must be a positive multiple of 32 on CUDA'
    x_grad_dtype = x.dtype if x_grad_dtype is None else x_grad_dtype
    assert out_grad.shape == x.shape and out_grad.device == x.device and out_grad.stride(-1) == 1
    assert out_grad.dtype in (torch.bfloat16, torch.float32) and x_grad_dtype in (torch.bfloat16, torch.float32)
    assert rstd.shape == (num_tokens * num_rms_groups,) and rstd.dtype == torch.float32 and rstd.device == x.device and rstd.is_contiguous()
    if residual_out_grad is not None:
        assert residual_out_grad.shape == x.shape and residual_out_grad.dtype == x.dtype and residual_out_grad.device == x.device
        assert residual_out_grad.stride(-1) == 1
    if weight is not None:
        assert weight.shape == (hidden,) and weight.device == x.device and weight.is_contiguous()
        assert weight.dtype in (torch.bfloat16, torch.float32)
    if weight_grad is not None:
        assert with_weight_grad
        assert weight_grad.shape == weight.shape and weight_grad.dtype in (torch.bfloat16, torch.float32) and weight_grad.device == x.device
        assert weight_grad.is_contiguous()

    if out is not None:
        assert out.shape == x.shape and out.dtype == x.dtype and out.device == x.device and out.is_contiguous()

    x_grad = torch.empty(x.shape, dtype=x_grad_dtype, device=x.device)
    weight_grad_initialized = weight_grad is not None
    if with_weight_grad and weight_grad is None:
        weight_grad = torch.empty_like(weight)
    if num_tokens == 0:
        if with_weight_grad and not weight_grad_initialized:
            weight_grad.zero_()
        return x_grad, weight_grad

    num_sms = get_num_vec_cores() if is_ascend() else get_num_sms()
    num_partials = num_sms if is_ascend() else _get_norm_backward_schedule(hidden, rms_group_size, num_sms)[0]
    weight_grad_partial = None if not with_weight_grad else torch.empty((num_partials, hidden), dtype=torch.float32, device=x.device)
    kernel_kwargs = dict(
        hidden=hidden,
        rms_group_size=rms_group_size,
        with_weight=weight is not None,
        with_residual_out_grad=residual_out_grad is not None,
        weight_grad_initialized=weight_grad_initialized,
        out_scale=out_scale,
        x_dtype=T.dtype(x.dtype),
        out_grad_dtype=T.dtype(out_grad.dtype),
        x_grad_dtype=T.dtype(x_grad.dtype),
        weight_dtype=T.dtype(x.dtype if weight is None else weight.dtype),
        weight_grad_dtype=T.dtype(weight_grad.dtype if weight_grad is not None else x.dtype),
        do_recompute=out is not None,
        with_weight_grad=with_weight_grad,
    )
    if is_ascend():
        from tile_kernels.quant.norm_backward_asc import get_norm_backward_kernel_asc

        kernel = get_norm_backward_kernel_asc(num_vec_cores=num_partials, **kernel_kwargs)
    else:
        for tensor in (x, out_grad, residual_out_grad):
            if tensor is not None:
                assert tensor.stride(0) % 16 == 0
        # Grid sync uses a cooperative launch, which TileLang cannot combine with its PDL launch path.
        kernel = get_norm_backward_kernel_cuda(num_sms=num_sms, **kernel_kwargs)
    args = (x, weight, out_grad, rstd, x_grad, weight_grad_partial, weight_grad, residual_out_grad, out)

    kernel(*args)

    return x_grad, weight_grad

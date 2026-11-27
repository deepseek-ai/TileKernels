from typing import Optional

import torch
from tilelang import language as T

from tile_kernels.config import get_num_vec_cores, get_pdl, get_token_alignment, is_ascend
from tile_kernels.quant.common import *
from tile_kernels.quant.swiglu_backward_cuda import get_swiglu_backward_and_per_token_cast_kernel_cuda


def swiglu_backward_and_per_token_cast_impl(
    x: Union[torch.Tensor, QuantTensor],
    act_x_grad: torch.Tensor,
    fmt: str,
    num_per_channels: Optional[int],
    psum_num_tokens_per_expert: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    routed_scaling_factor: Optional[float] = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    clamp_value: Optional[float] = None,
    do_recompute: bool = False,
) -> tuple:
    """Fuse SwiGLU backward pass with optional per-token FP8 quantization of the input gradient.

    Args:
        x: Forward input. For FP8 path: a ``QuantTensor`` ``(data, sf)`` where data has
            shape (num_expanded_tokens, hidden * 2) and sf has shape
            (num_expanded_tokens, hidden * 2 // num_per_channels).
            For BF16 path: a plain BF16 tensor of shape (num_expanded_tokens, hidden * 2).
        act_x_grad: BF16 or FP32 gradient of shape (num_expanded_tokens, hidden).
        fmt: Target format (``'e4m3'``, ``'bf16'`` or ``'fp32'``).
        num_per_channels: Number of channels in each scaling block. Must be 32. Required for FP8 path; ignored for BF16 path.
        psum_num_tokens_per_expert: Optional prefix-sum of per-expert token counts,
            shape (num_experts,) int32.
        topk_weights: Optional routing weights of shape (num_expanded_tokens,) fp32.
        routed_scaling_factor: Optional scalar multiplied into topk_weights and weight_grad.
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major sf.
        round_sf: Whether to round scaling factors to powers of two.
        use_packed_ue8m0: Whether to store sf in packed UE8M0 format.
        clamp_value: Optional clamp threshold for SwiGLU activations.
        do_recompute: Whether to do_recompute and return the forward activation output
            ``act_out``. When False, ``act_out`` is not computed or returned.

    Returns:
        A tuple in this fixed field order, where optional fields are present only
        when their condition holds:
            x_grad (always),
            (x_grad_fp8, x_grad_fp8_sf) (FP8 path only),
            weight_grad (only when topk_weights is not None),
            act_out (only when do_recompute is True).
    """
    assert fmt in ('e4m3', 'bf16', 'fp32')

    if fmt == 'e4m3':
        assert num_per_channels == 32, 'SwiGLU backward requires num_per_channels=32'
        x, x_sf = x
        assert x.dtype == torch.float8_e4m3fn
        assert x_sf.dim() == 2 and x_sf.is_contiguous()
    else:
        x_sf = None
        assert x.dtype in (torch.bfloat16, torch.float32)

    assert (x.dim() == 2 or x.dim() == 3) and x.is_contiguous()
    assert x.size(-1) % 2 == 0
    hidden = x.size(-1) // 2
    assert hidden % 64 == 0

    x = x.view(-1, hidden * 2)
    act_x_grad = act_x_grad.view(-1, hidden)
    num_expanded_tokens = x.size(0)
    if is_ascend() and fmt == 'e4m3':
        assert num_expanded_tokens % 128 == 0, f'Ascend FP8 SwiGLU backward requires num_expanded_tokens % 128 == 0, got {num_expanded_tokens}'

    if fmt == 'e4m3':
        assert hidden % num_per_channels == 0
        assert x_sf.shape == (num_expanded_tokens, 2 * hidden // num_per_channels)
    assert act_x_grad.dtype in (torch.bfloat16, torch.float32)
    assert act_x_grad.shape == (num_expanded_tokens, hidden)

    if topk_weights is not None:
        assert topk_weights.dim() == 1 and topk_weights.is_contiguous()
        assert topk_weights.dtype == torch.float32
        assert topk_weights.shape == (num_expanded_tokens,)

    if psum_num_tokens_per_expert is not None:
        assert psum_num_tokens_per_expert.dim() == 1
        assert psum_num_tokens_per_expert.dtype == torch.int32
        assert psum_num_tokens_per_expert.is_contiguous()
    num_experts = psum_num_tokens_per_expert.shape[0] if psum_num_tokens_per_expert is not None else 0

    if routed_scaling_factor is not None:
        assert topk_weights is not None, 'routed_scaling_factor requires topk_weights'

    out_config = get_cast_output_config(
        fmt,
        (1, num_per_channels if num_per_channels is not None else 1),
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )

    # Get kernel implement
    if is_ascend():
        from tile_kernels.quant.swiglu_backward_asc import get_swiglu_backward_kernel_asc

        kernel = get_swiglu_backward_kernel_asc(
            hidden,
            num_experts,
            get_token_alignment(),
            topk_weights is not None,
            routed_scaling_factor is not None,
            clamp_value is not None,
            x_dtype=T.dtype(x.dtype),
            act_x_grad_dtype=T.dtype(act_x_grad.dtype),
            out_config=out_config,
            do_recompute=do_recompute,
            num_vec_cores=get_num_vec_cores(),
        )
    else:
        kernel = get_swiglu_backward_and_per_token_cast_kernel_cuda(
            hidden,
            num_experts,
            get_token_alignment(),
            topk_weights is not None,
            routed_scaling_factor is not None,
            clamp_value is not None,
            x_dtype=T.dtype(x.dtype),
            act_x_grad_dtype=T.dtype(act_x_grad.dtype),
            out_config=out_config,
            do_recompute=do_recompute,
            use_pdl=get_pdl(),
        )

    # Allocate output and launch.
    # their dtype follows the target format (e.g. bf16 or fp32).
    out_dtype = torch.bfloat16 if out_config.with_sf else out_config.torch_dtype
    out = torch.empty((num_expanded_tokens, hidden), dtype=out_dtype, device=act_x_grad.device) if do_recompute else None
    x_grad = torch.empty((num_expanded_tokens, hidden * 2), dtype=out_dtype, device=act_x_grad.device)
    weight_grad = torch.empty((num_expanded_tokens,), dtype=torch.float32, device=x.device) if topk_weights is not None else None

    if out_config.with_sf:
        x_grad_fp8 = torch.empty((num_expanded_tokens, hidden * 2), dtype=x.dtype, device=x.device)
        x_grad_fp8_sf = alloc_scaling_factors((num_expanded_tokens, hidden * 2), out_config, device=x.device)
    else:
        x_grad_fp8 = None
        x_grad_fp8_sf = None

    clamp_value = 0 if clamp_value is None else clamp_value
    routed_scaling_factor = routed_scaling_factor if routed_scaling_factor is not None else 1.0
    if num_expanded_tokens > 0:
        kernel(
            x,
            x_sf,
            act_x_grad,
            topk_weights,
            routed_scaling_factor,
            psum_num_tokens_per_expert,
            out,
            x_grad_fp8,
            x_grad_fp8_sf,
            x_grad,
            weight_grad,
            clamp_value,
        )

    # Field order: x_grad (always), then the optional fields in order:
    # (x_grad_fp8, x_grad_fp8_sf), weight_grad, act_out.
    result = [x_grad]
    if out_config.with_sf:
        x_grad_fp8_sf = cast_epilogue(x_grad_fp8_sf, out_config)
        result.append((x_grad_fp8, x_grad_fp8_sf))
    if topk_weights is not None:
        result.append(weight_grad)
    if do_recompute:
        result.append(out)
    return tuple(result)


def swiglu_backward_and_per_token_cast(
    x: QuantTensor,
    act_x_grad: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    psum_num_tokens_per_expert: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    routed_scaling_factor: Optional[float] = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    clamp_value: Optional[float] = None,
    do_recompute: bool = False,
) -> tuple:
    """Fuse SwiGLU backward pass with per-token FP8 quantization of the input gradient.

    Args:
        x: FP8 forward input as a ``QuantTensor`` ``(data, sf)`` where data has
            shape (num_expanded_tokens, hidden * 2) and sf has shape
            (num_expanded_tokens, hidden * 2 // num_per_channels).
            Note: when ``psum_num_tokens_per_expert`` is provided, padding slots
            must be 0.
        act_x_grad: BF16 gradient of shape (num_expanded_tokens, hidden).
            Note: when ``psum_num_tokens_per_expert`` is provided, padding slots
            must be 0.
        fmt: Target output format (``'e4m3'``).
        num_per_channels: Number of channels in each scaling block. Must be 32.
        topk_weights: Optional routing weights of shape (num_expanded_tokens,) fp32.
            When None, no per-row weight scaling is applied and weight_grad is
            not produced.
            Note: when ``psum_num_tokens_per_expert`` is provided, padding slots
            must be 0.
        routed_scaling_factor: Optional scalar multiplied into topk_weights and weight_grad.
        psum_num_tokens_per_expert: Optional prefix-sum of per-expert token counts,
            shape (num_experts,) int32. Expert e occupies expanded slots
            [align(psum[e-1], alignment), psum[e]); the range
            [psum[e-1], align(psum[e-1], alignment)) is padding. Pass None when
            the input has no expert layout (single contiguous segment).
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major sf.
        round_sf: Whether to round scaling factors to powers of two.
        use_packed_ue8m0: Whether to store sf in packed UE8M0 format.
        clamp_value: Optional clamp threshold for SwiGLU activations.
        do_recompute: Whether to do_recompute and return the forward activation output
            ``act_out``. When False, ``act_out`` is not computed or returned.

    Returns:
        A tuple in this fixed field order, where optional fields are present only
        when their condition holds:
            x_grad (always),
            (x_grad_fp8, x_grad_fp8_sf) (always present on this FP8 path),
            weight_grad (only when topk_weights is not None; 1D of shape
                (num_expanded_tokens,) with padding slots explicitly zeroed),
            act_out (only when do_recompute is True).
    """
    return swiglu_backward_and_per_token_cast_impl(
        x,
        act_x_grad,
        fmt,
        num_per_channels,
        psum_num_tokens_per_expert,
        topk_weights,
        routed_scaling_factor,
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
        clamp_value,
        do_recompute,
    )


def swiglu_backward(
    x: torch.Tensor,
    act_x_grad: torch.Tensor,
    fmt: str,
    psum_num_tokens_per_expert: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    routed_scaling_factor: Optional[float] = None,
    clamp_value: Optional[float] = None,
    do_recompute: bool = False,
) -> tuple:
    """Fuse SwiGLU backward pass without quantization.

    Args:
        x: BF16 or FP32 forward input of shape (num_expanded_tokens, hidden * 2).
            Note: when ``psum_num_tokens_per_expert`` is provided, padding slots
            must be 0.
        act_x_grad: BF16 or FP32 gradient of shape (num_expanded_tokens, hidden).
            Note: when ``psum_num_tokens_per_expert`` is provided, padding slots
            must be 0.
        fmt: Target output format (``'bf16'`` or ``'fp32'``).
        psum_num_tokens_per_expert: Optional prefix-sum of per-expert token counts,
            shape (num_experts,) int32. Expert e occupies expanded slots
            [align(psum[e-1], alignment), psum[e]); the range
            [psum[e-1], align(psum[e-1], alignment)) is padding. Pass None when
            the input has no expert layout (single contiguous segment).
        topk_weights: Optional routing weights of shape (num_expanded_tokens,) fp32.
            When None, no per-row weight scaling is applied and weight_grad is
            not produced.
            Note: when ``psum_num_tokens_per_expert`` is provided, padding slots
            must be 0.
        routed_scaling_factor: Optional scalar multiplied into topk_weights and weight_grad.
        clamp_value: Optional clamp threshold for SwiGLU activations.
        do_recompute: Whether to do_recompute and return the forward activation output
            ``act_out``. When False, ``act_out`` is not computed or returned.

    Returns:
        A tuple in this fixed field order, where optional fields are present only
        when their condition holds:
            x_grad (always),
            weight_grad (only when topk_weights is not None; 1D of shape
                (num_expanded_tokens,) with padding slots explicitly zeroed),
            act_out (only when do_recompute is True).
    """
    assert fmt in ('bf16', 'fp32')
    assert x.dtype in (torch.float32, torch.bfloat16) and x.dtype == act_x_grad.dtype
    return swiglu_backward_and_per_token_cast_impl(
        x,
        act_x_grad,
        fmt,
        None,
        psum_num_tokens_per_expert,
        topk_weights,
        routed_scaling_factor=routed_scaling_factor,
        clamp_value=clamp_value,
        do_recompute=do_recompute,
    )

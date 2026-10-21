from typing import Optional

import torch
from tilelang import language as T

from tile_kernels.config import get_num_sms, get_num_vec_cores, get_pdl, get_token_alignment, is_ascend
from tile_kernels.quant.common import *
from tile_kernels.quant.swiglu_forward_cuda import get_swiglu_forward_and_per_token_cast_kernel_cuda


def swiglu_forward_and_per_token_cast_impl(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: Optional[int],
    psum_num_tokens_per_expert: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    routed_scaling_factor: Optional[float] = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    clamp_value: Optional[float] = None,
    clamped_count: Optional[torch.Tensor] = None,
) -> Union[QuantTensor, torch.Tensor]:
    assert x.dim() == 2 and x.is_contiguous()
    num_expanded_tokens, hidden = x.shape
    assert hidden % 2 == 0
    hidden = hidden // 2
    if psum_num_tokens_per_expert is not None:
        assert psum_num_tokens_per_expert.dim() == 1
    num_experts = psum_num_tokens_per_expert.shape[0] if psum_num_tokens_per_expert is not None else 0

    if clamped_count is not None:
        assert clamp_value is not None
        assert clamped_count.dim() == 1
        assert clamped_count.shape[0] == 4
        assert num_expanded_tokens * ceil_div(hidden, 32) <= 0x7FFFFFFF, 'input is too large for int32 clamp counts'

    if routed_scaling_factor is not None:
        assert topk_weights is not None, 'routed_scaling_factor requires topk_weights'

    # Swiglu forward : hidden -> hidden // 2
    assert (fmt, num_per_channels) in [('e4m3', 128), ('e4m3', 32), ('bf16', None), ('fp32', None)]

    out_config = get_cast_output_config(
        fmt, (1, num_per_channels if num_per_channels != None else 1), use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0
    )

    # Get kernel implement
    if is_ascend():
        from tile_kernels.quant.swiglu_forward_asc import get_swiglu_forward_kernel_asc

        kernel = get_swiglu_forward_kernel_asc(
            hidden,
            num_experts,
            get_token_alignment(),
            topk_weights is not None,
            routed_scaling_factor is not None,
            clamp_value,
            clamped_count is not None,
            in_dtype=T.dtype(x.dtype),
            out_config=out_config,
            num_vec_cores=get_num_vec_cores(),
        )
    else:
        kernel = get_swiglu_forward_and_per_token_cast_kernel_cuda(
            hidden,
            num_experts,
            get_token_alignment(),
            topk_weights is not None,
            routed_scaling_factor is not None,
            clamp_value,
            clamped_count is not None,
            in_dtype=T.dtype(x.dtype),
            out_config=out_config,
            num_sms=get_num_sms(),
            use_pdl=get_pdl(),
        )

    # Allocate output and launch
    out = torch.empty((num_expanded_tokens, hidden), dtype=out_config.torch_dtype, device=x.device)
    out_sf = alloc_scaling_factors((num_expanded_tokens, hidden), out_config, device=x.device) if out_config.with_sf else None

    if num_expanded_tokens > 0:
        routed_scaling_factor = routed_scaling_factor if routed_scaling_factor is not None else 1.0
        kernel(x, out, out_sf, psum_num_tokens_per_expert, topk_weights, routed_scaling_factor, clamped_count)

    if out_config.with_sf:
        out_sf = cast_epilogue(out_sf, out_config)
        return out, out_sf
    else:
        return out


def swiglu_forward_and_per_token_cast(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    psum_num_tokens_per_expert: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    routed_scaling_factor: Optional[float] = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    clamp_value: Optional[float] = None,
    clamped_count: Optional[torch.Tensor] = None,
) -> QuantTensor:
    """Fuse SwiGLU forward"""
    return swiglu_forward_and_per_token_cast_impl(
        x,
        fmt,
        num_per_channels,
        psum_num_tokens_per_expert,
        topk_weights,
        routed_scaling_factor,
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
        clamp_value,
        clamped_count,
    )


def swiglu_forward(
    x: torch.Tensor,
    fmt: str,
    psum_num_tokens_per_expert: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    routed_scaling_factor: Optional[float] = None,
    clamp_value: Optional[float] = None,
    clamped_count: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert fmt in ('fp32', 'bf16'), 'swiglu_forward only supports fp32 or bf16 output'
    return swiglu_forward_and_per_token_cast_impl(
        x,
        fmt,
        None,
        psum_num_tokens_per_expert,
        topk_weights,
        routed_scaling_factor=routed_scaling_factor,
        clamp_value=clamp_value,
        clamped_count=clamped_count,
    )

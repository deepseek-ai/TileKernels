import torch
from typing import Optional

from tile_kernels.config import get_num_vec_cores, is_ascend
from tile_kernels.quant.common import *
from tile_kernels.quant.per_token_cast_cuda import get_per_token_cast_kernel_cuda


def per_token_cast_impl(
    x: Union[torch.Tensor, QuantTensor],
    fmt: str,
    num_per_channels: int,
    x_block_size: Optional[tuple[int, int]] = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    use_e4m3_sf: bool = False,
    sf_only: bool = False,
    stochastic_cast: bool = False,
    sf: Optional[torch.Tensor] = None,
    out_x: Optional[torch.Tensor] = None,
    out_sf: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, QuantTensor]:
    SMALL_BATCH_THRESHOLD = 4096
    assert fmt in ('e4m3', 'e2m1')

    x, x_sf, in_config = get_cast_input_and_config(x, x_block_size)
    assert num_per_channels in (16, 32), 'num_per_channels must be 16 or 32'
    assert x_block_size in (None, (32, 32))
    out_config = get_cast_output_config(fmt, (1, num_per_channels), use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0, use_e4m3_sf)

    num_tokens, hidden = x.shape
    hidden = get_logical_hidden(hidden, x.dtype)
    assert hidden % 64 == 0
    assert not use_e4m3_sf or not is_ascend(), 'E4M3 SF is only supported on CUDA'
    assert sf is None or out_sf is None
    assert sf is None or (not use_tma_aligned_col_major_sf and not use_packed_ue8m0), 'Cast-only mode only supports row-major, non-packed SF'

    # Get kernel implement
    if is_ascend():
        from tile_kernels.quant.per_token_cast_asc import get_per_token_cast_kernel_asc

        kernel = get_per_token_cast_kernel_asc(
            hidden=hidden,
            token_stride=get_logical_hidden(x.stride(0), x.dtype),
            in_config=in_config,
            out_config=out_config,
            sf_only=sf_only,
            cast_only=(sf is not None),
            small_batch=(stochastic_cast and num_tokens <= SMALL_BATCH_THRESHOLD),
            stochastic_cast=stochastic_cast,
            num_vec_cores=get_num_vec_cores(),
        )
    else:
        kernel = get_per_token_cast_kernel_cuda(
            hidden=hidden,
            token_stride=get_logical_hidden(x.stride(0), x.dtype),
            in_config=in_config,
            out_config=out_config,
            sf_only=sf_only,
            cast_only=(sf is not None),
            small_batch=num_tokens <= SMALL_BATCH_THRESHOLD,
            stochastic_cast=stochastic_cast,
            use_pdl=False,
        )  # PDL regression: >=2us slower, keep PDL off.

    # Allocate output and launch
    if out_x is None:
        out_x = torch.empty((num_tokens, get_physical_hidden(hidden, out_config.torch_dtype)), dtype=out_config.torch_dtype, device=x.device)
    if out_sf is None:
        out_sf = (
            sf
            if sf is not None
            else alloc_scaling_factors(
                (num_tokens, hidden),
                out_config=out_config,
                device=x.device,
            )
        )
    else:
        sf_shape = get_sf_shape((num_tokens, hidden), out_config)
        assert not use_tma_aligned_col_major_sf, 'Not supported yet'
        if use_packed_ue8m0:
            out_sf = out_sf.view(dtype=torch.uint8)
        assert out_sf.shape == sf_shape

    if num_tokens > 0:
        kernel(x, x_sf, out_x, out_sf)

    # Make corrected SF tensor
    out_sf = cast_epilogue(out_sf, out_config)

    if sf_only:
        return out_sf

    if sf is not None:
        return out_x

    return out_x, out_sf


def per_token_cast(
    x: Union[torch.Tensor, QuantTensor],
    fmt: str,
    num_per_channels: int,
    x_block_size: Optional[tuple[int, int]] = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    use_e4m3_sf: bool = False,
    stochastic_cast: bool = False,
    out: Optional[QuantTensor] = None,
) -> QuantTensor:
    """Cast a matrix to FP8/FP4 with per-token (row-wise) scaling factors.

    Args:
        x: Input 2D tensor or a tuple of 2D tensor and 2D scaling factors.
        fmt: Target quantized format (``'e4m3'`` or ``'e2m1'``).
        num_per_channels: Number of channels in each scaling block.
        x_block_size: Input scaling block size for pre-quantized inputs.
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major sf factors.
        round_sf: Whether to round scaling factors to powers of two.
        use_packed_ue8m0: Whether to use packed UE8M0 format for sf factors.
        use_e4m3_sf: Whether to store sf factors in E4M3 format.
        stochastic_cast: Whether to use stochastic rounding for the narrowing cast.
        out: Optional pre-allocated QuantTensor (data, sf).

    Returns:
        A tuple ``(out, out_sf)`` with quantized output and sf factors.
    """
    out_x, out_sf = (None, None) if out is None else out
    return per_token_cast_impl(
        x,
        fmt,
        num_per_channels,
        x_block_size,
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
        use_e4m3_sf,
        stochastic_cast=stochastic_cast,
        out_x=out_x,
        out_sf=out_sf,
    )


def per_token_cast_with_sf_only(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    use_e4m3_sf: bool = False,
    out_sf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cast a matrix to FP8/FP4 with per-token (row-wise) scaling, returning only the sf factors.

    Args:
        x: Input 2D tensor of shape (num_tokens, hidden).
        fmt: Target quantized format (``'e4m3'`` or ``'e2m1'``).
        num_per_channels: Number of channels in each scaling block.
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major sf factors.
        round_sf: Whether to round scaling factors to powers of two.
        use_packed_ue8m0: Whether to use packed UE8M0 format for sf factors.
        use_e4m3_sf: Whether to store sf factors in E4M3 format.
        out_sf: Optional pre-allocated sf factors.

    Returns:
        Scale-factor tensor only (quantized output is discarded).
    """
    return per_token_cast_impl(
        x,
        fmt,
        num_per_channels,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
        use_e4m3_sf=use_e4m3_sf,
        sf_only=True,
        out_sf=out_sf,
    )


def per_token_cast_with_precomputed_sf(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    sf: torch.Tensor,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    use_e4m3_sf: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cast a matrix to FP8/FP4 with per-token (row-wise) scaling using precomputed sf factors.

    Args:
        x: Input 2D tensor of shape (num_tokens, hidden).
        fmt: Target quantized format (``'e4m3'`` or ``'e2m1'``).
        num_per_channels: Number of channels in each scaling block.
        sf: Pre-computed sf factors.
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major sf factors.
        round_sf: Whether to round scaling factors to powers of two.
        use_packed_ue8m0: Whether to use packed UE8M0 format for sf factors.
        use_e4m3_sf: Whether to store sf factors in E4M3 format.
        out: Optional pre-allocated output tensor.

    Returns:
        Casted tensor using the precomputed sf factors.
    """
    return per_token_cast_impl(
        x,
        fmt,
        num_per_channels,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
        use_e4m3_sf=use_e4m3_sf,
        sf=sf,
        out_x=out,
    )

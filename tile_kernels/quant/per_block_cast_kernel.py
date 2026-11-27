from typing import Optional, Union

import torch

from tile_kernels.config import get_num_vec_cores, get_pdl
from tile_kernels.quant.common import *
from tile_kernels.quant.per_block_cast_cuda import get_per_block_cast_kernel_cuda


def per_block_cast_impl(
    x: torch.Tensor,
    fmt: str,
    block_size: tuple[int, int],
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    sf_only: bool = False,
    sf: Optional[torch.Tensor] = None,
    out_x: Optional[torch.Tensor] = None,
    out_sf: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, QuantTensor]:
    """
        sf_only: If True, only compute and return sf factors.
        sf: Pre-computed sf factors; if provided, cast-only mode is used.

    Returns:
        Scale-factor tensor when ``sf_only=True``, casted tensor when ``sf``
        is provided, or a tuple ``(out, out_sf)`` with quantized output and sf factors.
    """
    assert block_size == (32, 32), 'Per-block cast requires token block 32'
    assert x.is_contiguous() and x.dim() == 2
    num_tokens, hidden = x.shape

    assert num_tokens % 32 == 0
    assert hidden % 128 == 0
    assert sf is None or out_sf is None
    if sf is not None:
        assert not sf_only and not use_tma_aligned_col_major_sf and not use_packed_ue8m0

    x, _, in_config = get_cast_input_and_config(x, None)
    out_config = get_cast_output_config(fmt, block_size, use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0)

    if is_ascend():
        from tile_kernels.quant.per_block_cast_asc import get_per_block_cast_kernel_asc

        kernel = get_per_block_cast_kernel_asc(
            hidden=hidden,
            in_config=in_config,
            out_config=out_config,
            sf_only=sf_only,
            cast_only=(sf is not None),
            num_vec_cores=get_num_vec_cores(),
        )
    else:
        kernel = get_per_block_cast_kernel_cuda(
            hidden=hidden,
            in_config=in_config,
            out_config=out_config,
            sf_only=sf_only,
            cast_only=(sf is not None),
            use_pdl=get_pdl(),
        )

    if out_x is None and not sf_only:
        out_x = torch.empty((num_tokens, get_physical_hidden(hidden, out_config.torch_dtype)), dtype=out_config.torch_dtype, device=x.device)
    elif out_x is not None:
        # per_block_cast writes out in 16-byte vectorized chunks, so its base address must be 16-byte aligned.
        assert out_x.data_ptr() % 16 == 0, f'out must be 16-byte aligned, got address {out_x.data_ptr():#x}'
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
        assert not use_tma_aligned_col_major_sf, 'pre-allocated out_sf does not support TMA-aligned column-major layout yet'
        if use_packed_ue8m0:
            packed_dtype = get_packed_ue8m0_torch_dtype()
            assert out_sf.dtype == packed_dtype, f'{out_sf.dtype=} expected {packed_dtype}'
            out_sf = out_sf.view(dtype=torch.uint8)
        assert out_sf.shape == sf_shape, f'{out_sf.shape=} expected {sf_shape}'

    if num_tokens > 0:
        kernel(x, out_x, out_sf)

    out_sf = cast_epilogue(out_sf, out_config)

    if sf_only:
        return out_sf

    if sf is not None:
        return out_x

    return out_x, out_sf


def per_block_cast(
    x: torch.Tensor,
    fmt: str,
    block_size: tuple[int, int],
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    out: Optional[QuantTensor] = None,
) -> QuantTensor:
    """Cast a matrix to FP8/FP4 with per-block scaling factors.

    Args:
        x: Input 2D contiguous tensor of shape (num_tokens, hidden).
        fmt: Target quantized format (``'e4m3'`` or ``'e2m1'``).
        block_size: Scaling block size as (num_per_tokens, num_per_channels).
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major sf factors.
        round_sf: Whether to round scaling factors to powers of two.
        use_packed_ue8m0: Whether to use packed UE8M0 format for sf factors.
        out: Optional pre-allocated QuantTensor (data, sf).

    Returns:
       Tuple of quantized output and sf factors.
    """
    out_x, out_sf = (None, None) if out is None else out
    return per_block_cast_impl(
        x,
        fmt,
        block_size,
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
        out_x=out_x,
        out_sf=out_sf,
    )


def per_block_cast_with_sf_only(
    x: torch.Tensor,
    fmt: str,
    block_size: tuple[int, int],
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    out_sf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cast a matrix to FP8/FP4, only output the scaling factors.

    Args:
        x: Input 2D contiguous tensor of shape (num_tokens, hidden).
        fmt: Target quantized format (``'e4m3'`` or ``'e2m1'``).
        block_size: Scaling block size as (num_per_tokens, num_per_channels).
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major sf factors.
        round_sf: Whether to round scaling factors to powers of two.
        use_packed_ue8m0: Whether to use packed UE8M0 format for sf factors.
        out_sf: Optional pre-allocated sf factors.

    Returns:
        Scaling factors only
    """
    return per_block_cast_impl(
        x,
        fmt,
        block_size,
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
        sf_only=True,
        out_sf=out_sf,
    )


def per_block_cast_with_precomputed_sf(
    x: torch.Tensor,
    fmt: str,
    block_size: tuple[int, int],
    sf: torch.Tensor,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cast a matrix to FP8/FP4 with per-block scaling using precomputed sf factors.

    Args:
        x: Input 2D contiguous tensor of shape (num_tokens, hidden).
        fmt: Target quantized format (``'e4m3'`` or ``'e2m1'``).
        block_size: Scaling block size as (num_per_tokens, num_per_channels).
        sf: Pre-computed sf factors.
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major sf factors.
        round_sf: Whether to round scaling factors to powers of two.
        use_packed_ue8m0: Whether to use packed UE8M0 format for sf factors.
        out: Optional pre-allocated output tensor.

    Returns:
        Casted tensor using the precomputed sf factors.
    """
    return per_block_cast_impl(
        x,
        fmt,
        block_size,
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
        sf=sf,
        out_x=out,
    )

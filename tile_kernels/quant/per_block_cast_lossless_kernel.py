import torch

from tile_kernels.config import get_num_vec_cores, get_pdl
from tile_kernels.quant.common import *
from tile_kernels.quant.per_block_cast_lossless_cuda import get_per_block_cast_lossless_kernel_cuda


def per_block_cast_lossless(
    x: QuantTensor,
    fmt: str,
    x_block_size: tuple[int, int],
    out_block_size: tuple[int, int],
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> QuantTensor:
    """Losslessly re-quantize an FP4 (E2M1) tensor to FP8 (E4M3) with new block sizes.

    Args:
        x: Quantized tensor pair ``(data, sf_factors)`` in E2M1 format.
        fmt: Target format (must be ``'e4m3'``).
        x_block_size: Input scaling block size as (num_per_tokens, num_per_channels).
        out_block_size: Output scaling block size as (num_per_tokens, num_per_channels).
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major sf factors.
        round_sf: Whether to round scaling factors to powers of two.
        use_packed_ue8m0: Whether to use packed UE8M0 format for sf factors.

    Returns:
        A tuple ``(out, out_sf)`` with FP8 output and sf-factor tensor.
    """
    assert x_block_size == (1, 32) and out_block_size == (32, 32), 'Lossless cast only supports in (1, 32) / out (32, 32)'
    x, x_sf, in_config = get_cast_input_and_config(x, x_block_size)

    assert fmt == 'e4m3' and x.dtype == torch.int8, 'lossless cast only supports e2m1 -> e4m3 input'
    assert x.is_contiguous() and x.dim() == 2
    num_tokens, hidden = x.shape
    hidden = get_logical_hidden(hidden, x.dtype)
    assert num_tokens % 32 == 0, f'lossless cast requires num_tokens % 32 == 0, got {num_tokens}'
    assert hidden % 128 == 0

    out_config = get_cast_output_config(fmt, out_block_size, use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0)

    if x.device.type == 'npu':
        from tile_kernels.quant.per_block_cast_lossless_asc import get_per_block_cast_lossless_kernel_asc

        # FP4 is stored as packed int8; reinterpret as the 2x-packed fp4 dtype the kernel expects.
        x = x.view(torch.float4_e2m1fn_x2)
        kernel = get_per_block_cast_lossless_kernel_asc(
            hidden=hidden,
            in_config=in_config,
            out_config=out_config,
            num_vec_cores=get_num_vec_cores(),
        )
    else:
        kernel = get_per_block_cast_lossless_kernel_cuda(
            hidden=hidden,
            token_stride=get_logical_hidden(x.stride(0), x.dtype),
            in_config=in_config,
            out_config=out_config,
            use_pdl=get_pdl(),
        )

    out = torch.empty((num_tokens, hidden), dtype=out_config.torch_dtype, device=x.device)
    out_sf = alloc_scaling_factors(
        (num_tokens, hidden),
        out_config=out_config,
        device=x.device,
    )

    if num_tokens > 0:
        kernel(x, x_sf, out, out_sf)

    out_sf = cast_epilogue(out_sf, out_config)

    return out, out_sf

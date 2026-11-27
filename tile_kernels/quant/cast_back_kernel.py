import torch
from tilelang import language as T

from tile_kernels.config import get_num_vec_cores, is_ascend
from tile_kernels.quant.common import *
from tile_kernels.quant.cast_back_cuda import get_cast_back_kernel_cuda


def cast_back(
    x: QuantTensor,
    fmt: str,
    x_block_size: tuple[int, int],
    x_special_fmt: Optional[str] = None,
) -> torch.Tensor:
    """Dequantize an FP8/FP4 tensor back to BF16 or FP32.

    Args:
        x: Quantized tensor pair ``(data, sf_factors)``.
        fmt: Target output format, either ``'bf16'`` or ``'fp32'``.
        x_block_size: Scaling block size as ``(num_per_tokens, num_per_channels)``.
        x_special_fmt: Reserved, must be ``None``.

    Returns:
        Dequantized tensor in the requested format.
    """
    assert x_block_size in ((1, 32), (1, 16), (32, 32), (32, 1))
    assert fmt in ('bf16', 'fp32')
    out_dtype = torch.bfloat16 if fmt == 'bf16' else torch.float32

    x, x_sf = x

    assert x_special_fmt is None, f'Unsupported special format: {x_special_fmt}'

    num_tokens, hidden = x.shape
    assert x_sf.dim() == 2
    assert x_sf.dtype in (torch.float32, torch.float8_e4m3fn, get_packed_ue8m0_torch_dtype())

    hidden = get_logical_hidden(hidden, x.dtype)
    assert hidden % 64 == 0

    x, x_sf, in_config = get_cast_input_and_config((x, x_sf), x_block_size)
    assert not (in_config.use_e4m3_sf and is_ascend()), 'E4M3 SF is only supported on CUDA'
    assert not (x_block_size[1] == 1 and in_config.use_tma_aligned_col_major_sf), (
        'cast_back does not support col-major scaling factors when num_per_channels == 1'
    )

    if x.device.type == 'npu':
        from tile_kernels.quant.cast_back_asc import get_cast_back_kernel_asc

        if in_config.use_packed_ue8m0:
            x_sf = x_sf.view(torch.uint16)
        kernel = get_cast_back_kernel_asc(
            hidden=hidden,
            in_config=in_config,
            out_dtype=T.dtype(out_dtype),
            num_vec_cores=get_num_vec_cores(),
        )
    else:
        kernel = get_cast_back_kernel_cuda(
            hidden=hidden,
            in_config=in_config,
            out_dtype=T.dtype(out_dtype),
            use_pdl=False,
        )  # PDL regression: >=2us slower, keep PDL off.

    out = torch.empty((num_tokens, hidden), dtype=out_dtype, device=x.device)
    if num_tokens > 0:
        kernel(x, x_sf, out)

    return out


def per_token_cast_back(
    x: QuantTensor,
    fmt: str,
    num_per_channels: int,
    x_special_fmt: Optional[str] = None,
) -> torch.Tensor:
    """Dequantize an FP8/FP4 tensor back to BF16 or FP32 with per-token scaling.

    Args:
        x: Quantized tensor pair ``(data, sf_factors)``.
        fmt: Target output format, either ``'bf16'`` or ``'fp32'``.
        num_per_channels: Number of channels per scaling block.
        x_special_fmt: Reserved, must be ``None``.

    Returns:
        Dequantized tensor in the requested format.
    """
    return cast_back(x, fmt, (1, num_per_channels), x_special_fmt=x_special_fmt)

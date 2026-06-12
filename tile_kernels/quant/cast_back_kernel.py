import math
import os

import tilelang
import torch
from tilelang import language as T

from tile_kernels.quant.common import *
from tile_kernels.utils import align
from .cast_back_e5m6_kernel import cast_back_e5m6


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True,
    },
)
def get_cast_back_kernel(
    hidden: int,
    in_config: CastInputConfig,
    out_dtype: T.dtype = T.bfloat16,
):
    num_threads = 128
    num_elems_per_block = 8192
    num_per_tokens, num_per_channels = in_config.sf_block

    if num_per_tokens == 1:
        # Default case: 1 * gcd(hidden, 8192)
        TILE_K = math.gcd(hidden, num_elems_per_block)

        # Replication optimization case: 1 * hidden
        if hidden <= num_elems_per_block:
            TILE_K = align(hidden, num_threads * (2 if in_config.dtype == T.float4_e2m1fn else 1))

        TILE_M = num_elems_per_block // TILE_K

        # When TILE_M is small, set TILE_M = 1 to avoid predicated loads for better performance
        if TILE_M <= 3:
            TILE_M = 1
    else:
        # Per block cast case: 128 * 64
        TILE_M, TILE_K = 128, 64

    # Runtime symbols
    num_tokens = T.dynamic('num_tokens')
    sf_shape = get_sf_shape((num_tokens, hidden), in_config)

    sf_stride = T.dynamic('sf_stride')

    @T.prim_func
    def cast_back_kernel(
        x: T.Tensor[(num_tokens, hidden), in_config.dtype],
        x_sf: T.StridedTensor[sf_shape, (sf_stride, 1), in_config.sf_dtype],
        out: T.Tensor[(num_tokens, hidden), out_dtype],
    ):
        with T.Kernel(T.ceildiv(num_tokens, TILE_M), T.ceildiv(hidden, TILE_K), threads=num_threads) as (pid_token, pid_hidden):
            x_shared = T.alloc_shared((TILE_M, TILE_K), in_config.dtype)
            sf_shared = T.alloc_shared((T.ceildiv(TILE_M, num_per_tokens), T.ceildiv(TILE_K, num_per_channels)), T.float32)
            out_fragment = T.alloc_fragment((TILE_M, TILE_K), out_dtype)

            T.copy(x[pid_token * TILE_M, pid_hidden * TILE_K], x_shared, disable_tma=True)
            num_sf_token_blocks = T.ceildiv(num_tokens, num_per_tokens)
            num_sf_channel_blocks = T.ceildiv(hidden, num_per_channels)
            for i, j in T.Parallel(T.ceildiv(TILE_M, num_per_tokens), T.ceildiv(TILE_K, num_per_channels)):
                token_index = pid_token * TILE_M // num_per_tokens + i
                channel_index = pid_hidden * TILE_K // num_per_channels + j
                if token_index < num_sf_token_blocks and channel_index < num_sf_channel_blocks:
                    sf = load_sf(x_sf, token_index, channel_index, in_config)
                    sf_shared[i, j] = transform_sf(sf, in_config)
                else:
                    sf_shared[i, j] = 0

            for i, j in T.Parallel(TILE_M, TILE_K):
                out_fragment[i, j] = x_shared[i, j] * sf_shared[i // num_per_tokens, j // num_per_channels]

            T.copy(out_fragment, out[pid_token * TILE_M, pid_hidden * TILE_K])

    return cast_back_kernel


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
        x_special_fmt: Optional special format identifier (e.g. ``'e5m6'``).

    Returns:
        Dequantized tensor in the requested format.
    """
    assert fmt in ('bf16', 'fp32')
    out_dtype = torch.bfloat16 if fmt == 'bf16' else torch.float32

    x, x_sf = x

    # Dispatch e5m6 packed format (stored as uint8 after forward cast)
    if x_special_fmt == 'e5m6':
        assert x.dtype == torch.uint8
        return cast_back_e5m6((x, x_sf), fmt, x_block_size)
    assert x_special_fmt is None, f'Unsupported special format: {x_special_fmt}'

    num_tokens, hidden = x.shape
    assert x_sf.dim() == 2
    assert x_sf.dtype in (torch.int32, torch.float32)

    hidden = get_logical_hidden(hidden, x.dtype)
    assert hidden % 64 == 0

    x, x_sf, in_config = get_cast_input_and_config((x, x_sf), x_block_size)

    kernel = get_cast_back_kernel(hidden=hidden, in_config=in_config, out_dtype=T.dtype(out_dtype))

    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

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
        x_special_fmt: Optional special format identifier (e.g. ``'e5m6'``).

    Returns:
        Dequantized tensor in the requested format.
    """
    return cast_back(x, fmt, (1, num_per_channels), x_special_fmt=x_special_fmt)

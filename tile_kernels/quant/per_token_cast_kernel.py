import os
import torch
import tilelang
import math
from typing import Optional
from tilelang import language as T

from tile_kernels.utils import align
from tile_kernels.quant.common import *
from .per_token_cast_to_e5m6_kernel import per_token_cast_to_e5m6


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True,
    },
)
def get_per_token_cast_kernel(
    hidden: int,
    token_stride: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
    sf_only: bool = False,
    cast_only: bool = False,
):
    num_threads = 128
    num_elems_per_thread = 32
    num_elems_per_block = num_threads * num_elems_per_thread
    num_per_channels = out_config.sf_block[1]

    if hidden == num_per_channels:
        assert not in_config.with_sf and not cast_only and not sf_only
        # Must ensure each thread has elements as multiple of 2 for FP4
        block_k = align(hidden, num_threads * (2 if out_config.dtype == T.float4_e2m1fn else 1))
        num_per_channels = block_k
    else:
        block_k = max(num_per_channels, math.gcd(num_elems_per_block, hidden))

    assert block_k % num_per_channels == 0
    block_m = 1 if num_elems_per_block % block_k != 0 else num_elems_per_block // block_k
    num_groups = block_k // num_per_channels

    num_vectorize = min(get_best_vectorize_size(in_config.dtype), math.gcd(block_m * block_k // num_threads, 32))
    if in_config.with_sf:
        assert not cast_only and not sf_only
        assert block_k % num_vectorize == 0
        assert num_per_channels >= num_vectorize, (
            f'num_per_channels ({num_per_channels}) must be >= num_vectorize ({num_vectorize}); on sm>=10 with float8, num_vectorize=32, so num_per_channels=16 would cause a zero-dimension reshape'
        )
        assert block_m % in_config.sf_block[0] == 0 or in_config.sf_block[0] % block_m == 0
        assert block_k % in_config.sf_block[1] == 0 or in_config.sf_block[1] % block_k == 0

    # Runtime symbols
    num_tokens = T.dynamic('num_tokens')
    in_sf_stride = T.dynamic('in_sf_stride')
    out_sf_stride = T.dynamic('out_sf_stride')
    x_sf_shape = get_sf_shape((num_tokens, hidden), in_config)
    sf_shape = get_sf_shape((num_tokens, hidden), out_config)

    def x_layout_fn(i: int, j: int):
        id = i * block_k + j
        return id // num_vectorize % num_threads, id // (num_vectorize * num_threads) * num_vectorize + id % num_vectorize

    @T.prim_func
    def per_token_cast_kernel(
        x: T.StridedTensor[(num_tokens, hidden), (token_stride, 1), in_config.dtype],
        x_sf: T.StridedTensor[x_sf_shape, (in_sf_stride, 1), in_config.sf_dtype],
        out: T.Tensor[(num_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[sf_shape, (out_sf_stride, 1), out_config.sf_dtype],
    ):
        with T.Kernel(T.ceildiv(num_tokens, block_m), T.ceildiv(hidden, block_k), threads=num_threads) as (pid_token, pid_hidden):
            x_fragment = T.alloc_fragment((block_m, block_k), in_config.dtype)
            sf_inv_fragment = T.alloc_fragment((block_m, num_groups), T.float32)
            out_shared = T.alloc_shared((block_m, block_k), out_config.dtype)

            T.annotate_layout({
                x_fragment: T.Fragment(
                    (block_m, block_k),
                    forward_fn=x_layout_fn,
                )
            })

            # Copy input into registers
            T.copy(x[pid_token * block_m, pid_hidden * block_k], x_fragment, disable_tma=True)

            if in_config.with_sf:
                num_sf_rows_per_block = T.ceildiv(block_m, in_config.sf_block[0])
                num_sf_cols_per_block = T.ceildiv(block_k, in_config.sf_block[1])
                x_sf_fragment = T.alloc_fragment((num_sf_rows_per_block, num_sf_cols_per_block), T.float32)
                for i, j in T.Parallel(num_sf_rows_per_block, num_sf_cols_per_block):
                    m_idx = pid_token * block_m // in_config.sf_block[0] + i
                    k_idx = pid_hidden * block_k // in_config.sf_block[1] + j
                    x_sf_fragment[i, j] = transform_sf(load_sf(x_sf, m_idx, k_idx, in_config), in_config)

                # Reduce stage 1, use half for reduction
                stage1_amax_fragment = T.alloc_fragment((block_m, block_k // num_vectorize), T.float16)
                x_stage1_fragment_reshaped = T.reshape(x_fragment, [block_m, block_k // num_vectorize, num_vectorize])
                T.reduce_absmax(x_stage1_fragment_reshaped, stage1_amax_fragment, dim=-1)

                # Apply scaling factor
                stage2_amax_fragment = T.alloc_fragment((block_m, block_k // num_vectorize), T.float32)
                T.clear(stage2_amax_fragment)
                for i, j in T.Parallel(block_m, block_k // num_vectorize):
                    stage2_amax_fragment[i, j] = (
                        T.float32(stage1_amax_fragment[i, j]) * x_sf_fragment[i // in_config.sf_block[0], j * num_vectorize // in_config.sf_block[1]]
                    )

                # Reduce stage 2, using float for reduction
                stage2_amax_fragment_reshaped = T.reshape(stage2_amax_fragment, [block_m, num_groups, block_k // num_vectorize // num_groups])
                T.reduce_max(stage2_amax_fragment_reshaped, sf_inv_fragment, dim=-1)

                for i, j in T.Parallel(block_m, num_groups):
                    sf, sf_inv = get_sf_and_inv(sf_inv_fragment[i, j], out_config)
                    # Store SF
                    m_idx = pid_token * block_m + i
                    k_idx = pid_hidden * num_groups + j
                    if m_idx < num_tokens:
                        store_sf(out_sf, sf, m_idx, k_idx, out_config)
                    sf_inv_fragment[i, j] = sf_inv

                # Store casted values
                if not sf_only:
                    for i, j in T.Parallel(block_m, block_k):
                        # Apply two multiplication at once to save the number of multiplication calculated
                        factor = x_sf_fragment[i // in_config.sf_block[0], j // in_config.sf_block[1]] * sf_inv_fragment[i, j // num_per_channels]
                        out_shared[i, j] = T.float32(x_fragment[i, j]) * factor

            else:
                if cast_only:
                    for i, j in T.Parallel(block_m, num_groups):
                        sf = load_sf(out_sf, pid_token * block_m + i, pid_hidden * num_groups + j, out_config)
                        sf_inv_fragment[i, j] = 1 / sf
                else:
                    amax_fragment = T.alloc_fragment((block_m, num_groups), T.float32)
                    x_fragment_reshaped = T.reshape(x_fragment, [block_m, num_groups, num_per_channels])
                    # Reduce SF
                    T.reduce_absmax(x_fragment_reshaped, amax_fragment, dim=2)
                    for i, j in T.Parallel(block_m, num_groups):
                        amax = T.cast(amax_fragment[i, j], T.float32)
                        sf, sf_inv = get_sf_and_inv(amax, out_config)

                        # Store SF
                        m_idx = pid_token * block_m + i
                        k_idx = pid_hidden * num_groups + j
                        if m_idx < num_tokens:
                            store_sf(out_sf, sf, m_idx, k_idx, out_config)
                        sf_inv_fragment[i, j] = sf_inv

                # Store casted values
                if not sf_only:
                    for i, j in T.Parallel(block_m, block_k):
                        out_shared[i, j] = x_fragment[i, j] * sf_inv_fragment[i, j // num_per_channels]

            if not sf_only:
                T.copy(out_shared, out[pid_token * block_m, pid_hidden * block_k], disable_tma=True)

    return per_token_cast_kernel


def per_token_cast_impl(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    x_block_size: Optional[tuple[int, int]] = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    sf_only: bool = False,
    sf: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, QuantTensor]:
    assert fmt in ('e5m6', 'e4m3', 'e2m1')
    if fmt == 'e5m6':
        assert x_block_size is None
        assert not sf_only
        assert sf is None
        return per_token_cast_to_e5m6(x, num_per_channels, use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0)

    x, x_sf, in_config = get_cast_input_and_config(x, x_block_size)
    out_config = get_cast_output_config(fmt, (1, num_per_channels), use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0)

    num_tokens, hidden = x.shape
    hidden = get_logical_hidden(hidden, x.dtype)
    assert num_per_channels in (16, 32, 64, 128) or (num_per_channels == hidden and hidden % 64 == 0)

    # Get kernel implement
    kernel = get_per_token_cast_kernel(
        hidden=hidden,
        token_stride=get_logical_hidden(x.stride(0), x.dtype),
        in_config=in_config,
        out_config=out_config,
        sf_only=sf_only,
        cast_only=(sf is not None),
    )

    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    # Allocate output and launch
    out = torch.empty((num_tokens, get_physical_hidden(hidden, out_config.torch_dtype)), dtype=out_config.torch_dtype, device=x.device)
    out_sf = (
        sf
        if sf is not None
        else alloc_scaling_factors(
            (num_tokens, hidden),
            out_config=out_config,
            device=x.device,
        )
    )
    if num_tokens > 0:
        kernel(x, x_sf, out, out_sf)

    # Make corrected SF tensor
    out_sf = cast_epilogue(out_sf, num_tokens, hidden, out_config)

    if sf_only:
        return out_sf

    if sf is not None:
        return out

    return out, out_sf


def per_token_cast(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    x_block_size: Optional[tuple[int, int]] = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> QuantTensor:
    """Cast a matrix to FP8/FP4 with per-token (row-wise) scaling factors.

    Args:
        x: Input 2D tensor of shape (num_tokens, hidden).
        fmt: Target quantized format (``'e5m6'``, ``'e4m3'``, or ``'e2m1'``).
        num_per_channels: Number of channels in each scaling block.
        x_block_size: Input scaling block size for pre-quantized inputs.
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major sf factors.
        round_sf: Whether to round scaling factors to powers of two.
        use_packed_ue8m0: Whether to use packed UE8M0 format for sf factors.

    Returns:
        A tuple ``(out, out_sf)`` with quantized output and sf factors.
    """
    return per_token_cast_impl(x, fmt, num_per_channels, x_block_size, use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0)


def per_token_cast_with_sf_only(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    x_block_size: Optional[tuple[int, int]] = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> torch.Tensor:
    """Cast a matrix to FP8/FP4 with per-token (row-wise) scaling, returning only the sf factors.

    Args:
        x: Input 2D tensor of shape (num_tokens, hidden).
        fmt: Target quantized format (``'e4m3'`` or ``'e2m1'``).
        num_per_channels: Number of channels in each scaling block.
        x_block_size: Input scaling block size for pre-quantized inputs.
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major sf factors.
        round_sf: Whether to round scaling factors to powers of two.
        use_packed_ue8m0: Whether to use packed UE8M0 format for sf factors.

    Returns:
        Scale-factor tensor only (quantized output is discarded).
    """
    return per_token_cast_impl(x, fmt, num_per_channels, x_block_size, use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0, sf_only=True)


def per_token_cast_with_precomputed_sf(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    sf: torch.Tensor,
    x_block_size: Optional[tuple[int, int]] = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> torch.Tensor:
    """Cast a matrix to FP8/FP4 with per-token (row-wise) scaling using precomputed sf factors.

    Args:
        x: Input 2D tensor of shape (num_tokens, hidden).
        fmt: Target quantized format (``'e4m3'`` or ``'e2m1'``).
        num_per_channels: Number of channels in each scaling block.
        sf: Pre-computed sf factors.
        x_block_size: Input scaling block size for pre-quantized inputs.
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major sf factors.
        round_sf: Whether to round scaling factors to powers of two.
        use_packed_ue8m0: Whether to use packed UE8M0 format for sf factors.

    Returns:
        Casted tensor using the precomputed sf factors.
    """
    return per_token_cast_impl(
        x, fmt, num_per_channels, x_block_size, use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0, sf_only=False, sf=sf
    )

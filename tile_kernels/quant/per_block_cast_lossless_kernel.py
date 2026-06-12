import os
import torch
import tilelang
from tilelang import language as T

from tile_kernels.utils import is_power_of_two
from tile_kernels.quant.common import *


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_per_block_cast_lossless_kernel(
    hidden: int,
    token_stride: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
):
    """Fp4 -> Fp8 Lossless cast
    Weight Shape = In Dim * Out Dim
    Input: 1*32 Quant UE8M0 Scale Inv E2M1 Type
    Output: 128*128/1*128 Quant UE8M0 Scale Inv E4M3 Type
    Note: Since the latter may not be fully compatible with the former, the current feasible approach is:
    max E2M1 = 2^2 * (1 + 0.5) = 11 1
    max E4M3 = 2^8 * (1 + 0.75) = 1111 110
    min E2M1 = 2^-1 * 1 = 00 1
    min E4M3 = 2^-6 * 1 = 0001 000
    Set the Fp8 Scale Inv to Max(Fp4 Scale Inv) / 2^6, and obtain the updated Scale Inv for each block: Fp4 Scale Inv / (Max(FP4 Scale Inv) / 2^6), with an upper limit of 2^6 and a lower limit of 2^-5 (the lower limit requires asserting Max(Fp4 Scale Inv)/Min(Fp4 Scale Inv) <= 2^11). Place the additional offset of Scale Inv into the exponent bits of Fp8.
    """
    num_threads = 256
    num_elements_per_block = 8192

    assert in_config.dtype == T.float4_e2m1fn and out_config.dtype == T.float8_e4m3fn, 'lossless mode only supports e2m1 -> e4m3 conversion currently'
    assert in_config.with_sf, 'lossless mode requires both input and output scaling factors'
    assert is_power_of_two(in_config.sf_block[1]) and is_power_of_two(out_config.sf_block[1]), 'block_k must be power of 2 for lossless mode'
    assert out_config.sf_block[0] % in_config.sf_block[0] == 0 and out_config.sf_block[1] % in_config.sf_block[1] == 0, (
        'Output block size must be multiple of input block size'
    )

    block_m = max(out_config.sf_block[0], 32)
    block_k = max(out_config.sf_block[1], num_elements_per_block // block_m)
    assert block_m % out_config.sf_block[0] == 0
    assert block_k % out_config.sf_block[1] == 0

    num_in_sf_per_block_m = block_m // in_config.sf_block[0]
    num_in_sf_per_block_k = block_k // in_config.sf_block[1]
    num_out_sf_per_block_m = block_m // out_config.sf_block[0]
    num_out_sf_per_block_k = block_k // out_config.sf_block[1]
    num_in_sf_per_out_sf_m = out_config.sf_block[0] // in_config.sf_block[0]
    num_in_sf_per_out_sf_k = out_config.sf_block[1] // in_config.sf_block[1]
    num_in_sf_per_out_sf = num_in_sf_per_out_sf_m * num_in_sf_per_out_sf_k

    # Runtime symbols
    num_tokens = T.dynamic('num_tokens')
    in_sf_shape = get_sf_shape((num_tokens, hidden), in_config)
    out_sf_shape = get_sf_shape((num_tokens, hidden), out_config)
    in_sf_stride = T.dynamic('in_sf_stride')
    out_sf_stride = T.dynamic('out_sf_stride')

    @T.macro
    def transform_sf_to_uint32(sf, sf_dtype):
        # Scaling factor must be positive
        if sf_dtype == T.float32:
            return T.reinterpret(sf, T.uint32) >> 23
        return T.uint32(sf)

    @T.macro
    def transform_sf_to_fp32(sf):
        return T.reinterpret(T.uint32(sf) << 23, T.float32)

    @T.prim_func
    def per_block_cast_lossless_kernel(
        x: T.StridedTensor[(num_tokens, hidden), (token_stride, 1), in_config.dtype],
        x_sf: T.StridedTensor[in_sf_shape, (in_sf_stride, 1), in_config.sf_dtype],
        out: T.Tensor[(num_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[out_sf_shape, (out_sf_stride, 1), out_config.sf_dtype],
    ):
        with T.Kernel(T.ceildiv(num_tokens, block_m), T.ceildiv(hidden, block_k), threads=num_threads) as (pid_token, pid_hidden):
            # Local buffers
            x_in_shared = T.alloc_shared((block_m, block_k), dtype=in_config.dtype)
            x_out_fragment = T.alloc_fragment((block_m, block_k), dtype=out_config.dtype)
            x_sf_fragment = T.alloc_fragment((num_in_sf_per_block_m, num_in_sf_per_block_k), dtype=in_config.sf_dtype)

            # Load x to fragment
            T.copy(x[pid_token * block_m : (pid_token + 1) * block_m, pid_hidden * block_k : (pid_hidden + 1) * block_k], x_in_shared, disable_tma=True)

            # Load scaling factor of x to fragment
            T.fill(x_sf_fragment, 0)
            num_in_sf_blocks_m = T.ceildiv(num_tokens, in_config.sf_block[0])
            num_in_sf_blocks_k = T.ceildiv(hidden, in_config.sf_block[1])
            for i, j in T.Parallel(num_in_sf_per_block_m, num_in_sf_per_block_k):
                m_idx = pid_token * block_m // in_config.sf_block[0] + i
                k_idx = pid_hidden * block_k // in_config.sf_block[1] + j
                if m_idx < num_in_sf_blocks_m and k_idx < num_in_sf_blocks_k:
                    x_sf_fragment[i, j] = load_sf(x_sf, m_idx, k_idx, in_config)

            # Alloc fragments
            x_sf_uint32_fragment = T.alloc_fragment((num_in_sf_per_block_m, num_in_sf_per_block_k), T.uint32)
            x_sf_uint32_fragment_reshaped = T.alloc_fragment((num_out_sf_per_block_m, num_out_sf_per_block_k, num_in_sf_per_out_sf), T.uint32)
            out_sf_uint32_fragment = T.alloc_fragment((num_out_sf_per_block_m, num_out_sf_per_block_k), T.uint32)

            T.fill(x_sf_uint32_fragment, 0)
            T.fill(x_sf_uint32_fragment_reshaped, 0)
            # Reshape input scaling factors to match output scaling factor layout
            for i, j in T.Parallel(num_in_sf_per_block_m, num_in_sf_per_block_k):
                out_sf_m_idx = i // num_in_sf_per_out_sf_m
                out_sf_k_idx = j // num_in_sf_per_out_sf_k
                in_sf_idx = (i % num_in_sf_per_out_sf_m) * num_in_sf_per_out_sf_k + (j % num_in_sf_per_out_sf_k)
                # Ensure out of bound scaling factor do not affect the result when the tensor is smaller than the block size.
                if i * in_config.sf_block[0] + pid_token * block_m < num_tokens and j * in_config.sf_block[1] + pid_hidden * block_k < hidden:
                    x_sf_uint32_fragment[i, j] = transform_sf_to_uint32(x_sf_fragment[i, j], in_config.sf_dtype)
                    x_sf_uint32_fragment_reshaped[out_sf_m_idx, out_sf_k_idx, in_sf_idx] = x_sf_uint32_fragment[i, j]

            # Get the max value of each block of scaling factors
            T.reduce_max(x_sf_uint32_fragment_reshaped, out_sf_uint32_fragment, dim=2)

            # Divide the output scaling factor by 2^6
            for i, j in T.Parallel(num_out_sf_per_block_m, num_out_sf_per_block_k):
                # Saturated subtraction to avoid overflow
                out_sf_uint32_fragment[i, j] = T.if_then_else(out_sf_uint32_fragment[i, j] >= 6, out_sf_uint32_fragment[i, j] - 6, 0)

            # Update the input scaling factor of each
            for i, j in T.Parallel(num_in_sf_per_block_m, num_in_sf_per_block_k):
                out_sf_m_idx_2 = i // num_in_sf_per_out_sf_m
                out_sf_k_idx_2 = j // num_in_sf_per_out_sf_k
                # Ensure out of bound scaling factor do not affect the result when the tensor is smaller than the block size.
                if i * in_config.sf_block[0] + pid_token * block_m < num_tokens and j * in_config.sf_block[1] + pid_hidden * block_k < hidden:
                    # Ensure the cast is lossless
                    T.device_assert(x_sf_uint32_fragment[i, j] + 5 >= out_sf_uint32_fragment[out_sf_m_idx_2, out_sf_k_idx_2])
                    # Allow the scaling factor overflow
                    x_sf_uint32_fragment[i, j] = x_sf_uint32_fragment[i, j] - out_sf_uint32_fragment[out_sf_m_idx_2, out_sf_k_idx_2] + 127

            # Multiply the scaling factor to the input tensor
            for i, j in T.Parallel(block_m, block_k):
                m_idx_2 = i // in_config.sf_block[0]
                k_idx_2 = j // in_config.sf_block[1]
                sf = transform_sf_to_fp32(x_sf_uint32_fragment[m_idx_2, k_idx_2])
                x_out_fragment[i, j] = T.cast(T.float32(x_in_shared[i, j]) * sf, out_config.dtype)

            # Store scaling factor back to global memory
            num_out_sf_blocks_m = T.ceildiv(num_tokens, out_config.sf_block[0])
            num_out_sf_blocks_k = T.ceildiv(hidden, out_config.sf_block[1])
            for i, j in T.Parallel(num_out_sf_per_block_m, num_out_sf_per_block_k):
                sf_m_idx = pid_token * num_out_sf_per_block_m + i
                sf_k_idx = pid_hidden * num_out_sf_per_block_k + j
                if sf_m_idx < num_out_sf_blocks_m and sf_k_idx < num_out_sf_blocks_k:
                    if out_config.use_packed_ue8m0:
                        sf = T.uint8(out_sf_uint32_fragment[i, j])
                    else:
                        sf = transform_sf_to_fp32(out_sf_uint32_fragment[i, j])
                    store_sf(out_sf, sf, sf_m_idx, sf_k_idx, out_config)

            T.copy(x_out_fragment, out[pid_token * block_m: (pid_token + 1) * block_m, pid_hidden * block_k: (pid_hidden + 1) * block_k])

    return per_block_cast_lossless_kernel


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
    x, x_sf, in_config = get_cast_input_and_config(x, x_block_size)

    assert fmt == 'e4m3' and x.dtype == torch.int8, 'lossless cast only supports e2m1 -> e4m3 input'
    assert x.is_contiguous() and x.dim() == 2
    num_tokens, hidden = x.shape
    hidden = get_logical_hidden(hidden, x.dtype)
    assert hidden % 32 == 0

    out_config = get_cast_output_config(fmt, out_block_size, use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0)

    kernel = get_per_block_cast_lossless_kernel(
        hidden=hidden,
        token_stride=get_logical_hidden(x.stride(0), x.dtype),
        in_config=in_config,
        out_config=out_config,
    )

    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    out = torch.empty((num_tokens, hidden), dtype=out_config.torch_dtype, device=x.device)
    out_sf = alloc_scaling_factors(
        (num_tokens, hidden),
        out_config=out_config,
        device=x.device,
    )

    if num_tokens > 0:
        kernel(x, x_sf, out, out_sf)

    out_sf = cast_epilogue(out_sf, num_tokens, hidden, out_config)

    return out, out_sf

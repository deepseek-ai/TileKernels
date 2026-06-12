import os
import math
import torch
import tilelang
from tilelang import language as T

from tile_kernels.quant.common import *


@T.macro
def get_sf_and_inv_e5m6(amax: float, out_config: CastOutputConfig):
    # Clamp with min value
    clamped_amax = T.max(amax, out_config.clamp_min_value)

    max_value = 65024
    sf = T.alloc_var(T.float32)
    sf = clamped_amax / max_value
    if not out_config.round_sf:
        return sf, max_value / clamped_amax

    # Round into 2's power
    bits = T.reinterpret(sf, T.uint32)
    # amax >= 1e-4 ensures sign bit = 0 and bits != 0 (no denorm/zero).
    # `(bits - 1) >> 23 + 1` gives ceil(log2).
    exp_sf = ((bits - 1) >> 23) + 1 - 127
    sf_inv = T.reinterpret((127 - exp_sf) << 23, T.float32)
    if out_config.use_packed_ue8m0:
        return T.uint8(exp_sf + 127), sf_inv
    else:
        return T.reinterpret((127 + exp_sf) << 23, T.float32), sf_inv


@T.macro
def float_to_e5m6(
    x: T.LocalBuffer[(8, ), T.float32],
    out: T.LocalBuffer[(3, ), T.uint32],
):
    half_u16 = T.alloc_local((8, ), T.uint16)
    value_half = T.alloc_var(T.float16)

    kCutBits = T.uint32(0b11111111111111111)
    kThreshold = T.uint32(0b10000000000000000)

    for i in T.unroll(8):
        value_half = T.call_extern(T.float16, '__float2half_rz', x[i])
        half_u16[i] = T.reinterpret(value_half, T.uint16)
        value_u32 = T.reinterpret(x[i], T.uint32)
        remain_bits = value_u32 & kCutBits
        half_u16[i] = half_u16[i] >> 4
        cond = ((half_u16[i] & T.uint16(1)) + remain_bits > kThreshold)
        half_u16[i] = half_u16[i] + T.cast(cond, T.uint16)

    out[0] = (T.cast(half_u16[0] << 20, T.uint32) |
              T.cast(half_u16[1] << 8, T.uint32)  |
              T.cast(half_u16[2] >> 4, T.uint32))

    out[1] = (T.cast(half_u16[2] << 28, T.uint32) |
              T.cast(half_u16[3] << 16, T.uint32) |
              T.cast(half_u16[4] << 4, T.uint32)  |
              T.cast(half_u16[5] >> 8, T.uint32))

    out[2] = (T.cast(half_u16[5] << 24, T.uint32) |
              T.cast(half_u16[6] << 12, T.uint32) |
              T.cast(half_u16[7], T.uint32))


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    },
)
def get_per_token_cast_to_e5m6_kernel(
    hidden: int,
    token_stride: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
):
    num_threads = 128
    num_elems_per_thread = 32
    num_elems_per_block = num_threads * num_elems_per_thread

    num_per_channels = out_config.sf_block[1]
    assert hidden == num_per_channels
    assert not in_config.with_sf

    if hidden == num_per_channels:
        # Must ensure each thread has elements as multiple of 8
        block_k = align(hidden, num_threads * 8)
        num_per_channels = block_k
    else:
        block_k = max(num_per_channels, math.gcd(num_elems_per_block, hidden))

    assert block_k % num_per_channels == 0
    block_m = 1 if num_elems_per_block % block_k != 0 else num_elems_per_block // block_k
    num_groups = block_k // num_per_channels

    # Runtime symbols
    num_tokens = T.dynamic('num_tokens')
    out_sf_stride = T.dynamic('out_sf_stride')
    scale_shape = get_sf_shape((num_tokens, hidden), out_config)


    def out_forward_fn(i: int, j: int):
        elems = i * block_k + j
        return elems // 8 % num_threads, elems % 8 + elems // (8 * num_threads) * 8


    @T.prim_func
    def per_token_cast_to_e5m6_kernel(
        x: T.StridedTensor[(num_tokens, hidden), (token_stride, 1), in_config.dtype],
        out: T.Tensor[(num_tokens, hidden // 8 * 3), out_config.dtype],
        out_sf: T.StridedTensor[scale_shape, (out_sf_stride, 1), out_config.sf_dtype],
    ):
        with T.Kernel(T.ceildiv(num_tokens, block_m), T.ceildiv(hidden, block_k), threads=num_threads) as (pid_token, pid_hidden):
            x_fragment = T.alloc_fragment((block_m, block_k), in_config.dtype)
            sf_inv_fragment = T.alloc_fragment((block_m, num_groups), T.float32)
            out_fragment = T.alloc_fragment((block_m, block_k), T.float32)

            # Copy input into registers
            T.copy(x[pid_token * block_m, pid_hidden * block_k], x_fragment)

            amax_fragment = T.alloc_fragment((block_m, num_groups), T.float32)
            x_fragment_reshaped = T.reshape(x_fragment, [block_m, num_groups, num_per_channels])
            # Reduce SF
            T.reduce_absmax(x_fragment_reshaped, amax_fragment, dim=2)
            for i, j in T.Parallel(block_m, num_groups):
                amax = T.cast(amax_fragment[i, j], T.float32)
                sf, sf_inv = get_sf_and_inv_e5m6(amax, out_config)

                # Store SF
                m_idx = pid_token * block_m + i
                k_idx = pid_hidden * num_groups + j
                if m_idx < num_tokens:
                    store_sf(out_sf, sf, m_idx, k_idx, out_config)
                sf_inv_fragment[i, j] = sf_inv

            T.annotate_layout({
                out_fragment: T.Fragment(
                    (block_m, block_k),
                    forward_fn=out_forward_fn,
                ),
            })

            for i, j in T.Parallel(block_m, block_k):
                out_fragment[i, j] = x_fragment[i, j] * sf_inv_fragment[i, j // num_per_channels]

            in_local = T.alloc_local((8, ), T.float32)
            out_local = T.alloc_local((3, ), T.uint32)

            for x, y in T.Parallel(block_m, block_k // 8):
                for j in T.serial(8):
                    in_local[j] = out_fragment[x, y * 8 + j]
                float_to_e5m6(in_local, out_local)
                m_idx = pid_token * block_m + x
                k_idx = pid_hidden * block_k + y * 8
                if m_idx < num_tokens and k_idx < hidden:
                    for j in T.serial(3):
                        out[m_idx, (k_idx // 8) * 3 + j] = out_local[j]

    return per_token_cast_to_e5m6_kernel


def per_token_cast_to_e5m6(
    x: torch.Tensor,
    num_per_channels: int,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> QuantTensor:
    """Cast a matrix to E5M6 (12-bit truncated half-precision) with per-token scaling factors.

    E5M6 packs each value into 12 bits (1-bit sign, 5-bit exponent, 6-bit mantissa)
    and stores 8 values as 3 uint32 words (96 bits). The output is returned as uint8.

    Args:
        x: Input 2D tensor of shape (num_tokens, hidden), BF16 or FP32.
        num_per_channels: Number of channels in each scaling block. Must equal ``hidden``.
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major sf factors.
        round_sf: Whether to round scaling factors to powers of two.
        use_packed_ue8m0: Whether to use packed UE8M0 format for sf factors.

    Returns:
        A tuple ``(out, out_sf)`` where ``out`` is a uint8 tensor of shape
        ``(num_tokens, hidden * 3 // 2)`` and ``out_sf`` is the sf-factor tensor.
    """
    # Checks
    assert x.dtype == torch.bfloat16 or x.dtype == torch.float32
    num_tokens, hidden = x.shape

    assert num_per_channels == hidden

    x, _, in_config = get_cast_input_and_config(x, None)
    out_config = get_cast_output_config('e5m6', (1, num_per_channels), use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0, 1e-4)

    # Get kernel implement
    kernel = get_per_token_cast_to_e5m6_kernel(
        hidden=hidden,
        token_stride=x.stride(0),
        in_config=in_config,
        out_config=out_config,
    )

    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    # Allocate output and launch
    out = torch.empty((num_tokens, hidden // 8 * 3), dtype=torch.uint32, device=x.device)
    out_sf = alloc_scaling_factors(
        (num_tokens, hidden),
        out_config=out_config,
        device=x.device,
    )
    if num_tokens > 0:
        kernel(x, out, out_sf)

    out = out.view(torch.uint8)
    out_sf = cast_epilogue(out_sf, num_tokens, hidden, out_config)

    return out, out_sf

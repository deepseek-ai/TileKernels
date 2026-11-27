import tilelang
from tilelang import language as T

from tile_kernels.utils import align, is_power_of_two
from tile_kernels.quant.common import *


# E2M1 -> E4M3 is a *bijection on the 16 code points* once the (power-of-two) rescale is
# folded in, so the whole "unpack fp4 -> widen to f32 -> multiply -> pack to e4m3" chain
# (2 conversion-pipe ops + 2 PRMT per element on sm_103) collapses to one PRMT + one mask
# per 4 elements, all on the ALU pipe:
#
#   nonzero code:  byte = MAG[|code|] + (d << 3)  |  (sign << 7)      d = in_sf_exp - out_sf_exp
#   code 0      :  byte = 0                                           (-0 -> 0x80)
#
# MAG = (e4m3(0), e4m3(.5), e4m3(1), e4m3(1.5), e4m3(2), e4m3(3), e4m3(4), e4m3(6))
#     = (0x00, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c)
# `d << 3` is exactly "add d to the e4m3 exponent field"; it cannot carry into the sign
# bit because 1 <= E+d <= 15 for every code (E in 6..9 for .5 .. 6, d in [-5, 6]) and the
# smallest table byte is 0x30 > 40 = max |d8|, so byte 0 of the table stays 0x00 (zero
# stays zero).  The `device_assert` below is the same one the fp32 path uses to guarantee
# d >= -5, i.e. that no code can round into the e4m3 subnormal range.
LOSSLESS_TRANSCODE_SOURCE = r"""
// 4 packed e2m1 nibbles (low 16 bits of `w`) -> 4 e4m3 bytes.  `sfb` is the per-block
// byte value x_sf_exp - out_sf_exp + 127 (same quantity the fp32 path reinterprets).
// The attributes are spelled out instead of using TL_DEVICE: imported sources are emitted
// before the `tl_templates/...` includes that define it.
__device__ __forceinline__ int tk_lossless_x4(unsigned short w, unsigned sfb) {
  int d8 = ((int)sfb - 127) * 8;
  unsigned ta = 0x3c383000u + (unsigned)(d8 * 0x01010100);  // e4m3(0, .5, 1, 1.5)
  unsigned tb = 0x4c484440u + (unsigned)(d8 * 0x01010101);  // e4m3(2, 3, 4, 6)
  unsigned mag = __byte_perm(ta, tb, (unsigned)w & 0x7777u);
  unsigned sgn = (__byte_perm((unsigned)w, (unsigned)(w >> 4), 0x5140u) & 0x08080808u) << 4;
  return (int)(mag | sgn);
}
"""


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_per_block_cast_lossless_kernel_cuda(
    hidden: int,
    token_stride: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
    use_pdl: bool = False,
):
    """Fp4 -> Fp8 Lossless cast
    Weight Shape = In Dim * Out Dim
    Input: 1*32 Quant UE8M0 Scale E2M1 Type
    Output: 128*128/1*128 Quant UE8M0 Scale E4M3 Type
    Scales are dequantization factors: decoded = quantized * scale.
    Note: Since the latter may not be fully compatible with the former, the current feasible approach is:
    max E2M1 = 2^2 * (1 + 0.5) = 11 1
    max E4M3 = 2^8 * (1 + 0.75) = 1111 110
    min E2M1 = 2^-1 * 1 = 00 1
    min E4M3 = 2^-6 * 1 = 0001 000
    Set the Fp8 Scale to Max(Fp4 Scale) / 2^6, and obtain the updated Scale for each block: Fp4 Scale / (Max(FP4 Scale) / 2^6), with an upper limit of 2^6 and a lower limit of 2^-5 (the lower limit requires asserting Max(Fp4 Scale)/Min(Fp4 Scale) <= 2^11). Place the additional offset of Scale into the exponent bits of Fp8.
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
    # Prevent block_k to be larger than hidden
    block_k = min(block_k, align(hidden, out_config.sf_block[1]))
    assert block_m % out_config.sf_block[0] == 0
    assert block_k % out_config.sf_block[1] == 0

    assert in_config.sf_block[1] % 4 == 0, 'lossless transcode needs sf_block_k % 4 == 0'
    assert hidden % 4 == 0
    num_quads = hidden // 4
    block_k_q = block_k // 4

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
            if use_pdl:
                T.pdl_sync()
            # Local buffers
            T.import_source(LOSSLESS_TRANSCODE_SOURCE)
            # Integer-lane views: 1 int16 of `x` holds 4 packed e2m1 nibbles, 1 int32 of `out`
            # holds 4 e4m3 bytes.  Keeping the view in the kernel (rather than asking the caller
            # for `.view(torch.int16)` tensors) lets both backends take the same argument types.
            x_q = T.view(x, (num_tokens, num_quads), T.int16)
            out_q = T.view(out, (num_tokens, num_quads), T.int32)
            x_in_shared = T.alloc_shared((block_m, block_k // 4), dtype=T.int16)
            x_out_fragment = T.alloc_fragment((block_m, block_k // 4), dtype=T.int32)
            x_sf_fragment = T.alloc_fragment((num_in_sf_per_block_m, num_in_sf_per_block_k), dtype=in_config.sf_dtype)

            # Load x to fragment
            T.copy(
                x_q[pid_token * block_m : (pid_token + 1) * block_m, pid_hidden * block_k_q : (pid_hidden + 1) * block_k_q],
                x_in_shared,
                disable_tma=True,
            )

            # Load scaling factor of x to fragment
            T.fill(x_sf_fragment, 0)
            for i, j in T.Parallel(num_in_sf_per_block_m, num_in_sf_per_block_k):
                m_idx = pid_token * block_m // in_config.sf_block[0] + i
                k_idx = pid_hidden * block_k // in_config.sf_block[1] + j
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

            # Rescale + transcode: 4 e2m1 -> 4 e4m3 per integer lane, ALU only.
            for i, j in T.Parallel(block_m, block_k // 4):
                m_idx_2 = i // in_config.sf_block[0]
                k_idx_2 = (j * 4) // in_config.sf_block[1]
                x_out_fragment[i, j] = T.call_extern(T.int32, 'tk_lossless_x4', x_in_shared[i, j], x_sf_uint32_fragment[m_idx_2, k_idx_2])

            # Store scaling factor back to global memory
            for i, j in T.Parallel(num_out_sf_per_block_m, num_out_sf_per_block_k):
                sf_m_idx = pid_token * num_out_sf_per_block_m + i
                sf_k_idx = pid_hidden * num_out_sf_per_block_k + j
                if out_config.use_packed_ue8m0:
                    sf = T.uint8(out_sf_uint32_fragment[i, j])
                else:
                    sf = transform_sf_to_fp32(out_sf_uint32_fragment[i, j])
                store_sf(out_sf, sf, sf_m_idx, sf_k_idx, out_config)

            T.copy(x_out_fragment, out_q[pid_token * block_m : (pid_token + 1) * block_m, pid_hidden * block_k_q : (pid_hidden + 1) * block_k_q])

            if use_pdl:
                T.pdl_trigger()

    return per_block_cast_lossless_kernel

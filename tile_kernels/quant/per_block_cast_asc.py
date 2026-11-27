import math
import tilelang

from tile_kernels.utils import align
from tile_kernels.quant.common import *

# Re-bind below the star import: tile_kernels.quant.common exports its own `T`.
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S


@tilelang.jit()
def get_per_block_cast_kernel_asc(
    hidden: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
    sf_only: bool,
    cast_only: bool,
    num_vec_cores: int,
):
    num_per_tokens, num_per_channels = out_config.sf_block
    assert (num_per_tokens, num_per_channels) == (32, 32)
    assert out_config.dtype in (T.float4_e2m1fn, T.float8_e4m3fn)

    block_k_limit = 512 if in_config.dtype == T.float32 else 1024
    block_k = align(math.gcd(block_k_limit, hidden), 128)
    block_m = num_per_tokens
    num_stages = 2

    num_sf_cols_per_block = block_k // num_per_channels
    pack_factor = get_packed_ue8m0_pack_factor()
    assert num_sf_cols_per_block % pack_factor == 0
    packed_sf_cols_per_block = num_sf_cols_per_block // pack_factor
    num_stored_sf_cols_per_block = packed_sf_cols_per_block if out_config.use_packed_ue8m0 else num_sf_cols_per_block
    sf_ub_dtype = T.uint16 if out_config.use_packed_ue8m0 else T.float32

    token_group = 4 if out_config.use_tma_aligned_col_major_sf and out_config.use_packed_ue8m0 else 1

    num_tokens = T.dynamic('num_tokens')
    sf_shape = get_sf_shape((num_tokens, hidden), out_config)
    sf_stride = T.dynamic('sf_stride')

    @T.macro
    def f32_pair_to_bf16(x0, x1, round_odd=False):  # round_odd=False means round_to_zero here
        # in this kernel, we ensure that round_odd=False is used only when the low 16bit of fp32 inputs are all zero.
        low, high = S.vdintlv(T.reinterpret(x0, 'uint16x128'), T.reinterpret(x1, 'uint16x128'))
        if round_odd:
            high = S.vor(high, S.vmins(low, 1))
        return T.reinterpret(high, 'bfloat16x128')

    @T.macro
    def pack_two_block_amax(amax_vec, m_low, m_high):
        a0 = S.vcmax(amax_vec, m_low)
        a1 = S.vcmax(amax_vec, m_high)
        return S.vsel(S.vdupv(a0), S.vdupv(a1), m_low)

    @T.macro
    def store_scale_pair(sf_ub, sf_row, sf_j, scale01):
        if out_config.use_packed_ue8m0:
            exponent_high = S.vdupv(scale01, pos='POS_HIGHEST')
            packed_exponents = S.vor(scale01, S.vshls(exponent_high, 8))
            if out_config.use_tma_aligned_col_major_sf:
                S.vsts(sf_ub[sf_j // pack_factor, sf_row], T.reinterpret(packed_exponents, 'uint16x128'), dist='ONEPT_B16')
            else:
                S.vsts(sf_ub[sf_row, sf_j // pack_factor], T.reinterpret(packed_exponents, 'uint16x128'), dist='ONEPT_B16')
        elif out_config.use_tma_aligned_col_major_sf:
            S.vsts(sf_ub[sf_j, sf_row], scale01, dist='ONEPT_B32')
            S.vsts(sf_ub[sf_j + 1, sf_row], S.vdupv(scale01, pos='POS_HIGHEST'), dist='ONEPT_B32')
        else:
            S.vsts(sf_ub[sf_row, sf_j], scale01, dist='ONEPT_B32')
            S.vsts(sf_ub[sf_row, sf_j + 1], S.vdupv(scale01, pos='POS_HIGHEST'), dist='ONEPT_B32')

    @T.macro
    def load_scale_pair(sf_ub, sf_j, m_low):
        scale0 = S.vld(sf_ub[0, sf_j], dist='BRC_B32')
        scale1 = S.vld(sf_ub[0, sf_j + 1], dist='BRC_B32')
        return S.vsel(scale0, scale1, m_low)

    @T.macro
    def quant_block(x_ub, sf_ub, out_ub, sf_row):
        with T.SimdVF():
            m_low = S.pset(32, 'PAT_VL32')
            m_high = S.vcmps(S.vci(0, T.int32), 31, op='gt')
            one_reg = S.vdup(1.0, T.float32)
            zero_bf16 = S.vdup(0.0, T.bfloat16)
            abs_mask = T.reinterpret(S.vdup(0x7FFF, T.uint16), 'bfloat16x128')

            for chunk in T.serial(block_k // 128):
                col = chunk * 128
                sf_col = chunk * 4  # 4 = 128 // num_per_channels
                inverse = S.alloc_local((2,), T.float32)
                if cast_only:
                    for half in T.unroll(2, explicit=True):
                        scale = load_scale_pair(sf_ub, sf_col + half * 2, m_low)
                        if out_config.round_sf:
                            exponent = S.vshrs(T.reinterpret(scale, 'uint32x64'), 23)
                            inverse_bits = S.vshls(S.vsub(S.vdup(254, T.uint32), exponent), 23)
                            inverse[half] = T.reinterpret(inverse_bits, 'float32x64')
                        else:
                            inverse[half] = S.vdiv(one_reg, scale)
                else:
                    amax = S.alloc_local((2,), T.float32)
                    if in_config.dtype == T.bfloat16:
                        amax_bf16 = S.alloc_var(T.bfloat16)
                        amax_bf16 = S.vdup(0.0, T.bfloat16)
                        for row in T.serial(num_per_tokens):
                            amax_bf16 = S.vmax(amax_bf16, S.vand(S.vld(x_ub[row, col]), abs_mask))
                        amax_low, amax_high = S.vintlv(zero_bf16, amax_bf16)
                        amax[0] = T.reinterpret(amax_low, 'float32x64')
                        amax[1] = T.reinterpret(amax_high, 'float32x64')
                    else:
                        for half in T.unroll(2, explicit=True):
                            amax[half] = S.vdup(0.0, T.float32)
                        for row in T.serial(num_per_tokens):
                            for half in T.unroll(2, explicit=True):
                                amax[half] = S.vmax(amax[half], S.vabs(S.vld(x_ub[row, col + half * 64])))
                    for half in T.unroll(2, explicit=True):
                        scale, inverse[half] = get_sf_and_inv_asc(pack_two_block_amax(amax[half], m_low, m_high), out_config)
                        store_scale_pair(sf_ub, sf_row, sf_col + half * 2, scale)

                if in_config.dtype == T.bfloat16 and out_config.round_sf:
                    sf_inv = f32_pair_to_bf16(inverse[0], inverse[1])

                if not sf_only:
                    for row in T.serial(num_per_tokens):
                        if in_config.dtype == T.bfloat16 and out_config.round_sf:
                            quantized_bf16 = S.vmul(S.vld(x_ub[row, col]), sf_inv)
                            if out_config.dtype == T.float8_e4m3fn:
                                interleaved_low, interleaved_high = S.vintlv(zero_bf16, quantized_bf16)
                                quantized_low = T.reinterpret(interleaved_low, 'float32x64')
                                quantized_high = T.reinterpret(interleaved_high, 'float32x64')
                        else:
                            if in_config.dtype == T.bfloat16:
                                interleaved_low, interleaved_high = S.vintlv(zero_bf16, S.vld(x_ub[row, col]))
                                input_low = T.reinterpret(interleaved_low, 'float32x64')
                                input_high = T.reinterpret(interleaved_high, 'float32x64')
                            else:
                                input_low = S.vld(x_ub[row, col])
                                input_high = S.vld(x_ub[row, col + 64])
                            quantized_low = S.vmul(input_low, inverse[0])
                            quantized_high = S.vmul(input_high, inverse[1])
                            if out_config.dtype == T.float4_e2m1fn:
                                quantized_bf16 = f32_pair_to_bf16(quantized_low, quantized_high, round_odd=True)

                        if out_config.dtype == T.float4_e2m1fn:
                            S.vsts(out_ub[row, col], S.vcvt(quantized_bf16, T.float4_e2m1fn), dist='PK4_B32')
                        else:
                            S.vsts(out_ub[row, col], S.vcvt(quantized_low, T.float8_e4m3fn), dist='PK4_B32')
                            S.vsts(out_ub[row, col + 64], S.vcvt(quantized_high, T.float8_e4m3fn), dist='PK4_B32')

    @T.prim_func
    def per_block_cast_kernel(
        x: T.Tensor[(num_tokens, hidden), in_config.dtype],
        out: T.Tensor[(num_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[sf_shape, (sf_stride, 1), out_config.sf_dtype],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            x_ub = T.alloc_shared((block_m, block_k), in_config.dtype)
            sf_cols = max(num_stored_sf_cols_per_block, 16 // token_group if out_config.use_packed_ue8m0 else 8)
            if out_config.use_tma_aligned_col_major_sf:
                # Packed SF tails use MTE Normal mode, which requires a 32-byte UB row stride.
                sf_rows = align(token_group, 32 // sf_ub_dtype.bytes) if out_config.use_packed_ue8m0 else token_group
                sf_ub = T.alloc_shared((sf_cols, sf_rows), sf_ub_dtype)
            else:
                sf_ub = T.alloc_shared((token_group, sf_cols), sf_ub_dtype)
            out_ub = T.alloc_shared((block_m, block_k), out_config.dtype)

            versions = {x_ub: num_stages, sf_ub: num_stages}
            if not sf_only:
                versions[out_ub] = num_stages
            T.annotate_buffer_versions(versions)

            for pid_x, pid_y in T.Persistent(
                [T.ceildiv(num_tokens, block_m * token_group), T.ceildiv(hidden, block_k)],
                num_vec_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
            ):
                sf_m_base = T.alloc_var(T.int32, init=pid_x * token_group)
                sf_k_base = T.alloc_var(T.int32, init=pid_y * num_sf_cols_per_block)
                stored_sf_k_base = T.alloc_var(T.int32, init=pid_y * num_stored_sf_cols_per_block)
                valid_tokens = T.alloc_var(T.int32, init=T.min(token_group, num_tokens // block_m - sf_m_base))
                for token_offset in T.serial(valid_tokens):
                    token_block = sf_m_base + token_offset
                    T.copy(x[token_block * block_m, pid_y * block_k], x_ub[0:block_m, 0:block_k])
                    if cast_only:
                        T.copy(out_sf[token_block, sf_k_base], sf_ub[0:1, 0:num_sf_cols_per_block])
                    quant_block(x_ub, sf_ub, out_ub, token_offset)
                    if not sf_only:
                        T.copy(out_ub[0:block_m, 0:block_k], out[token_block * block_m, pid_y * block_k])

                if not cast_only:
                    if out_config.use_packed_ue8m0:
                        output_sf = T.Tensor((sf_shape[0], sf_shape[1] // pack_factor), T.uint16, out_sf.data)
                    else:
                        output_sf = out_sf
                    if out_config.use_tma_aligned_col_major_sf:
                        T.copy(sf_ub[0:num_stored_sf_cols_per_block, 0:valid_tokens], output_sf[stored_sf_k_base, sf_m_base])
                    else:
                        T.copy(sf_ub[0:token_group, 0:num_stored_sf_cols_per_block], output_sf[sf_m_base, stored_sf_k_base])

    return per_block_cast_kernel

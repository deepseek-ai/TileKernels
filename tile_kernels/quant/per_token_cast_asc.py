import math

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from tile_kernels.quant.common import CastInputConfig, CastOutputConfig, get_packed_ue8m0_pack_factor, get_sf_shape, get_sf_and_inv_asc
from tile_kernels.quant.stochastic_rounding import stochastic_round_fp8_asc
from tile_kernels.utils import align


@tilelang.jit
def get_per_token_cast_kernel_asc(
    hidden: int,
    token_stride: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
    sf_only: bool,
    cast_only: bool,
    small_batch: bool,
    stochastic_cast: bool,
    num_vec_cores: int,
):
    num_per_channels = out_config.sf_block[1]
    assert not stochastic_cast or out_config.dtype == T.float8_e4m3fn, 'Ascend stochastic per_token_cast only supports E4M3 output'
    assert hidden % 128 == 0, 'Ascend per_token_cast hidden must be a multiple of 128'
    assert num_per_channels == 32, 'Ascend per_token_cast currently only supports 32 channels per scale'
    assert not in_config.with_sf or in_config.sf_block == (32, 32), 'Ascend per_token_cast expects input scale block size (32, 32)'
    pack_factor = get_packed_ue8m0_pack_factor()

    value_dtype = T.float32
    if (
        not stochastic_cast
        and hidden % 256 == 0
        and not cast_only
        and out_config.use_packed_ue8m0
        and (in_config.dtype == T.bfloat16 or in_config.with_sf and in_config.use_packed_ue8m0)
    ):
        value_dtype = T.bfloat16

    if value_dtype == T.bfloat16:
        if in_config.dtype == T.float8_e4m3fn:
            block_m, block_k_limit = 16, 2048
        else:
            block_m, block_k_limit = 32, 1024
    else:
        block_m, block_k_limit = 32, 512
        if stochastic_cast and hidden % 256 == 0 and not (in_config.with_sf and (small_batch or in_config.dtype == T.float8_e4m3fn)):
            block_m, block_k_limit = 16, 1024
    block_k = math.gcd(block_k_limit, hidden)
    if stochastic_cast:
        block_k = max(256, block_k)
    if in_config.with_sf:
        # Keep each tile within one input scale row. sf_block[0] < block_m cause bug
        block_m = math.gcd(block_m, in_config.sf_block[0])
    log2_block_m = block_m.bit_length() - 1
    num_stages = 2
    num_groups = block_k // num_per_channels
    num_hidden_tiles = T.ceildiv(hidden, block_k)
    sf_stride = 64

    num_tokens = T.dynamic('num_tokens')
    out_sf_stride = T.dynamic('out_sf_stride')
    sf_shape = get_sf_shape((num_tokens, hidden), out_config)
    x_sf_shape = get_sf_shape((num_tokens, hidden), in_config) if in_config.with_sf else (1, 1)

    @T.macro
    def load_packed_input_scale_pair(scales_ub, group, sf_shift, sf_exp_mask):
        packed = S.vld(scales_ub[group // pack_factor], dist='BRC_B16')
        packed_u32 = T.reinterpret(packed, 'uint32x64')
        scale_bits = S.vand(S.vshl(packed_u32, sf_shift), sf_exp_mask)
        return T.reinterpret(scale_bits, 'float32x64')

    @T.macro
    def init_transpose_indices(indices):
        with T.SimdVF():
            if out_config.use_packed_ue8m0:
                lane = S.vci(0, T.int16)
                output_row = S.vand(T.reinterpret(lane, 'uint16x128'), S.vdup(block_m - 1, T.uint16))
                group_in_vector = T.reinterpret(S.vshrs(lane, log2_block_m), 'uint16x128')
                source_indices = S.vadd(S.vmuls(output_row, sf_stride // pack_factor), group_in_vector)
                S.vsts(indices[0], source_indices, dist='NORM_B16')
            else:
                lane = S.vci(0, T.int32)
                output_row = S.vand(T.reinterpret(lane, 'uint32x64'), S.vdup(block_m - 1, T.uint32))
                group_in_vector = T.reinterpret(S.vshrs(lane, log2_block_m), 'uint32x64')
                source_indices = S.vadd(S.vmuls(output_row, sf_stride), group_in_vector)
                S.vsts(indices[0], source_indices, dist='NORM_B32')

    @T.macro
    def transpose_vectors(src, dst, indices, vector_lanes, num_output_groups, store_dist):
        with T.SimdVF():
            groups_per_vector = vector_lanes // block_m
            num_output_values = num_output_groups * block_m
            num_full_vectors = num_output_values // vector_lanes
            num_tail_values = num_output_values % vector_lanes
            source_indices = S.vld(indices[0])

            for output_vector in T.unroll(num_full_vectors, explicit=True):
                full_group_base = output_vector * groups_per_vector
                values = S.vgather2(src[0, full_group_base], source_indices)
                S.vsts(dst[full_group_base, 0], values, dist=store_dist)

            if num_tail_values > 0:
                tail_group_base = num_full_vectors * groups_per_vector
                tail_mask = S.pset(2048 // vector_lanes, f'PAT_VL{num_tail_values}')
                values = S.vgather2(src[0, tail_group_base], source_indices, tail_mask)
                S.vsts(dst[tail_group_base, 0], values, tail_mask, dist=store_dist, extent=num_tail_values)

    @T.macro
    def transpose_output_sf(src, dst, indices):
        if out_config.use_packed_ue8m0:
            src_u16 = T.Tensor((block_m, sf_stride // pack_factor), T.uint16, src.data)
            dst_u16 = T.Tensor((num_groups // pack_factor, block_m), T.uint16, dst.data)
            transpose_vectors(src_u16, dst_u16, indices, 128, num_groups // pack_factor, 'NORM_B16')
        else:
            transpose_vectors(src, dst, indices, 64, num_groups, 'NORM_B32')

    @T.macro
    def dequantize_input(x_ub, scales_ub, out_ub):
        with T.SimdVF():
            mask_low = S.pset(32, 'PAT_VL32')
            if in_config.use_packed_ue8m0:
                sf_shift = S.vsel(S.vdup(23, T.int32), S.vdup(15, T.int32), mask_low)
                sf_exp_mask = S.vdup(0x7F800000, T.uint32)
            for pair in T.serial(block_k // 128):
                col = pair * 128
                group = pair * 4
                if in_config.use_packed_ue8m0:
                    scale0 = load_packed_input_scale_pair(scales_ub, group, sf_shift, sf_exp_mask)
                    scale1 = load_packed_input_scale_pair(scales_ub, group + 2, sf_shift, sf_exp_mask)
                else:
                    scale0 = S.vsel(S.vld(scales_ub[group], dist='BRC_B32'), S.vld(scales_ub[group + 1], dist='BRC_B32'), mask_low)
                    scale1 = S.vsel(S.vld(scales_ub[group + 2], dist='BRC_B32'), S.vld(scales_ub[group + 3], dist='BRC_B32'), mask_low)
                if value_dtype == T.bfloat16:
                    _, scale_bf16 = S.vdintlv(T.reinterpret(scale0, 'bfloat16x128'), T.reinterpret(scale1, 'bfloat16x128'))
                for row in T.serial(block_m):
                    if value_dtype == T.bfloat16:
                        if in_config.dtype == T.float8_e4m3fn:
                            x0 = S.vcvt(S.vld(x_ub[row, col], dist='UNPK4_B8'), T.float32)
                            x1 = S.vcvt(S.vld(x_ub[row, col + 64], dist='UNPK4_B8'), T.float32)
                            _, x_bf16 = S.vdintlv(T.reinterpret(x0, 'bfloat16x128'), T.reinterpret(x1, 'bfloat16x128'))
                        else:
                            x_bf16 = S.vcvt(S.vld(x_ub[row, col], dist='UNPK4_B8'), T.bfloat16)
                        S.vsts(out_ub[row, col], S.vmul(x_bf16, scale_bf16), dist='NORM_B16')
                    else:
                        if in_config.dtype == T.float8_e4m3fn:
                            x0 = S.vcvt(S.vld(x_ub[row, col], dist='UNPK4_B8'), T.float32)
                            x1 = S.vcvt(S.vld(x_ub[row, col + 64], dist='UNPK4_B8'), T.float32)
                        else:
                            packed = S.vld(x_ub[row, col], dist='UNPK4_B8')
                            x_bf16 = S.vcvt(packed, T.bfloat16)
                            even = S.vcvt(x_bf16, T.float32, part=0)
                            odd = S.vcvt(x_bf16, T.float32, part=1)
                            x0, x1 = S.vintlv(even, odd)
                        S.vsts(out_ub[row, col], S.vmul(x0, scale0))
                        S.vsts(out_ub[row, col + 64], S.vmul(x1, scale1))

    @T.macro
    def quantize_bf16(x_ub, amax_ub, sf_inv_ub, sf_dense_ub, out_ub):
        with T.SimdVF():
            mask_vl8 = S.pset(32, 'PAT_VL8')
            abs_mask = S.vdup(0x7FFF, T.uint16)
            zero = S.vdup(0.0, T.bfloat16)
            for row in T.serial(block_m):
                for tile in T.serial(block_k // 256):
                    x0, x1 = S.vld2(x_ub[row, tile * 256], dist='DINTLV_B16')
                    abs_x0 = S.vand(T.reinterpret(x0, 'uint16x128'), abs_mask)
                    abs_x1 = S.vand(T.reinterpret(x1, 'uint16x128'), abs_mask)
                    maxima = S.vcgmax(S.vmax(abs_x0, abs_x1))
                    dense, _ = S.vintlv(zero, T.reinterpret(maxima, 'bfloat16x128'))
                    S.vsts(amax_ub[row, tile * 8], T.reinterpret(dense, 'float32x64'), mask_vl8, dist='NORM_B32', extent=8)
            S.mem_bar('VST_VLD')
            for row in T.serial(block_m):
                scale, inverse = get_sf_and_inv_asc(S.vld(amax_ub[row, 0]), out_config)
                S.vsts(sf_dense_ub[row, 0], T.reinterpret(scale, 'uint8x256'), dist='PK4_B32')
                S.vsts(sf_inv_ub[row, 0], S.vcvt(inverse, T.bfloat16), dist='PK_B32')
            if not sf_only:
                S.mem_bar('VST_VLD')
                for row in T.serial(block_m):
                    for tile in T.serial(block_k // 256):
                        col = tile * 256
                        x0, x1 = S.vld2(x_ub[row, col], dist='DINTLV_B16')
                        inverse = S.vld(sf_inv_ub[row, tile * 8], dist='E2B_B16')
                        quantized0 = S.vmul(x0, inverse)
                        quantized1 = S.vmul(x1, inverse)
                        if out_config.dtype == T.float4_e2m1fn:
                            c0, c1 = S.vintlv(quantized0, quantized1)
                            S.vsts(out_ub[row, col], S.vcvt(c0, T.float4_e2m1fn), dist='PK4_B32')
                            S.vsts(out_ub[row, col + 128], S.vcvt(c1, T.float4_e2m1fn), dist='PK4_B32')
                        else:
                            quantized_fp8_0 = S.vcvt(S.vcvt(quantized0, T.float32, part=0), T.float8_e4m3fn, part=0)
                            quantized_fp8_2 = S.vcvt(S.vcvt(quantized0, T.float32, part=1), T.float8_e4m3fn, part=2)
                            quantized_fp8_1 = S.vcvt(S.vcvt(quantized1, T.float32, part=0), T.float8_e4m3fn, part=1)
                            quantized_fp8_3 = S.vcvt(S.vcvt(quantized1, T.float32, part=1), T.float8_e4m3fn, part=3)
                            merged = S.vor(
                                S.vor(T.reinterpret(quantized_fp8_0, 'uint8x256'), T.reinterpret(quantized_fp8_2, 'uint8x256')),
                                S.vor(T.reinterpret(quantized_fp8_1, 'uint8x256'), T.reinterpret(quantized_fp8_3, 'uint8x256')),
                            )
                            S.vsts(out_ub[row, col], T.reinterpret(merged, 'float8_e4m3fnx256'), dist='NORM_B8')

    @T.macro
    def load_f32_pair(x_ub, row, col):
        if in_config.dtype == T.bfloat16:
            return S.vcvt(S.vld(x_ub[row, col], dist='UNPK_B16'), T.float32), S.vcvt(S.vld(x_ub[row, col + 64], dist='UNPK_B16'), T.float32)
        return S.vld(x_ub[row, col]), S.vld(x_ub[row, col + 64])

    @T.macro
    def load_f32_vectors(x_ub, row, col):
        # x0,x1,x2,x3 aligns 0,1,128,129 in the same col
        if in_config.dtype == T.bfloat16:
            x0 = S.vld(x_ub[row, col])
            x1 = S.vld(x_ub[row, col + 128])
            return (
                S.vcvt(x0, T.float32, part=0),
                S.vcvt(x0, T.float32, part=1),
                S.vcvt(x1, T.float32, part=0),
                S.vcvt(x1, T.float32, part=1),
            )
        x0, x1 = S.vld2(x_ub[row, col], dist='DINTLV_B32')
        x2, x3 = S.vld2(x_ub[row, col + 128], dist='DINTLV_B32')
        return x0, x1, x2, x3

    @T.macro
    def get_quad_offsets(base_idx):
        lanes = T.reinterpret(S.vci(0, T.int32), 'uint32x64')
        return S.vadds(lanes, base_idx)

    @T.macro
    def pad_tail(x_ub, sf_dense_ub):
        # Pad the last 128 columns of each token before computing scales and hash.
        zero = S.vdup(0.0, T.bfloat16 if in_config.dtype == T.bfloat16 else T.float32)
        if cast_only:
            mask_sf = S.pset(32, 'PAT_VL4')
        for row in T.serial(block_m):
            S.vsts(x_ub[row, hidden % block_k], zero, dist='NORM_B16' if in_config.dtype == T.bfloat16 else 'NORM_B32')
            if in_config.dtype != T.bfloat16:
                S.vsts(x_ub[row, hidden % block_k + 64], zero)
            if cast_only:
                scales = S.vsel(S.vld(sf_dense_ub[row, 0]), S.vdup(1.0, T.float32), mask_sf)
                S.vsts(sf_dense_ub[row, 0], scales)
        S.mem_bar('VST_VLD')

    @T.macro
    def quantize_f32(x_ub, amax_ub, sf_inv_ub, sf_dense_ub, out_ub, pid_hidden, step):
        with T.SimdVF():
            if step == 128:
                mask_low = S.pset(32, 'PAT_VL32')
            if hidden % block_k != 0 and pid_hidden == num_hidden_tiles - 1:
                pad_tail(x_ub, sf_dense_ub)
            if cast_only:
                one = S.vdup(1.0, T.float32)
                for row in T.serial(block_m):
                    S.vsts(sf_inv_ub[row, 0], S.vdiv(one, S.vld(sf_dense_ub[row, 0])))
            else:
                if step == 256:
                    mask_amax = S.pset(32, 'PAT_VL8')
                else:
                    mask_all = S.pset(32, 'PAT_ALL')
                    mask_high = S.pnot(mask_low, mask_all)
                for row in T.serial(block_m):
                    for tile in T.serial(block_k // step):
                        col = tile * step
                        group = tile * (step // 32)
                        if step == 256:
                            x0, x1, x2, x3 = load_f32_vectors(x_ub, row, col)
                            abs_x0, abs_x1 = S.vabs(x0), S.vabs(x1)
                            abs_x2, abs_x3 = S.vabs(x2), S.vabs(x3)
                            even, odd = S.vdintlv(S.vmax(abs_x0, abs_x1), S.vmax(abs_x2, abs_x3))
                            maxima = S.vcgmax(S.vmax(even, odd))
                            S.vsts(amax_ub[row, group], maxima, mask_amax, extent=8)
                        else:
                            x0, x1 = load_f32_pair(x_ub, row, col)
                            abs_x0, abs_x1 = S.vabs(x0), S.vabs(x1)
                            S.vsts(amax_ub[row, group], S.vcmax(abs_x0, mask_low), dist='ONEPT_B32')
                            S.vsts(amax_ub[row, group + 1], S.vcmax(abs_x0, mask_high), dist='ONEPT_B32')
                            S.vsts(amax_ub[row, group + 2], S.vcmax(abs_x1, mask_low), dist='ONEPT_B32')
                            S.vsts(amax_ub[row, group + 3], S.vcmax(abs_x1, mask_high), dist='ONEPT_B32')
                S.mem_bar('VST_VLD')
                for row in T.serial(block_m):
                    scale, inverse = get_sf_and_inv_asc(S.vld(amax_ub[row, 0]), out_config)
                    S.vsts(sf_inv_ub[row, 0], inverse)
                    if out_config.use_packed_ue8m0:
                        S.vsts(sf_dense_ub[row, 0], T.reinterpret(scale, 'uint8x256'), dist='PK4_B32')
                    else:
                        S.vsts(sf_dense_ub[row, 0], scale)
            if not sf_only:
                S.mem_bar('VST_VLD')
                if step == 256:
                    if stochastic_cast:
                        quad_offsets = get_quad_offsets(pid_hidden * (block_k // 4))
                    for tile in T.serial(block_k // 256):
                        col = tile * 256
                        group = tile * 8
                        if stochastic_cast:
                            offsets = S.vadds(quad_offsets, tile * 64)
                        for row in T.serial(block_m):
                            sf_inv = S.vld(sf_inv_ub[row, group], dist='E2B_B32')
                            sf_inv0, sf_inv1 = S.vintlv(sf_inv, sf_inv)
                            x0, x1, x2, x3 = load_f32_vectors(x_ub, row, col)
                            quantized = S.alloc_local((4,), T.float32)
                            quantized[0] = S.vmul(x0, sf_inv0)
                            quantized[1] = S.vmul(x1, sf_inv0)
                            quantized[2] = S.vmul(x2, sf_inv1)
                            quantized[3] = S.vmul(x3, sf_inv1)
                            if stochastic_cast:
                                rounded = stochastic_round_fp8_asc(quantized, offsets)
                            else:
                                quantized_fp8_0 = S.vcvt(quantized[0], T.float8_e4m3fn, part=0)
                                quantized_fp8_1 = S.vcvt(quantized[1], T.float8_e4m3fn, part=1)
                                quantized_fp8_2 = S.vcvt(quantized[2], T.float8_e4m3fn, part=0)
                                quantized_fp8_3 = S.vcvt(quantized[3], T.float8_e4m3fn, part=1)
                                packed0 = S.vor(T.reinterpret(quantized_fp8_0, 'uint16x128'), T.reinterpret(quantized_fp8_1, 'uint16x128'))
                                packed1 = S.vor(T.reinterpret(quantized_fp8_2, 'uint16x128'), T.reinterpret(quantized_fp8_3, 'uint16x128'))
                                packed, _ = S.vdintlv(packed0, packed1)
                                rounded = T.reinterpret(packed, 'float8_e4m3fnx256')
                            S.vsts(out_ub[row, col], rounded, dist='NORM_B8')
                else:
                    for row in T.serial(block_m):
                        for pair in T.serial(block_k // 128):
                            col = pair * 128
                            group = pair * 4
                            sf_inv0 = S.vld(sf_inv_ub[row, group], dist='BRC_B32')
                            sf_inv1 = S.vld(sf_inv_ub[row, group + 1], dist='BRC_B32')
                            sf_inv2 = S.vld(sf_inv_ub[row, group + 2], dist='BRC_B32')
                            sf_inv3 = S.vld(sf_inv_ub[row, group + 3], dist='BRC_B32')
                            x0, x1 = load_f32_pair(x_ub, row, col)
                            quantized0 = S.vmul(x0, S.vsel(sf_inv0, sf_inv1, mask_low))
                            quantized1 = S.vmul(x1, S.vsel(sf_inv2, sf_inv3, mask_low))
                            if out_config.dtype == T.float4_e2m1fn:
                                low, high = S.vdintlv(T.reinterpret(quantized0, 'uint16x128'), T.reinterpret(quantized1, 'uint16x128'))
                                packed = T.reinterpret(S.vor(high, S.vmins(low, 1)), 'bfloat16x128')
                                S.vsts(out_ub[row, col], S.vcvt(packed, T.float4_e2m1fn), dist='PK4_B32')
                            else:
                                S.vsts(out_ub[row, col], S.vcvt(quantized0, T.float8_e4m3fn), dist='PK4_B32')
                                S.vsts(out_ub[row, col + 64], S.vcvt(quantized1, T.float8_e4m3fn), dist='PK4_B32')

    @T.prim_func
    def per_token_cast_kernel(
        x: T.StridedTensor[(num_tokens, hidden), (token_stride, 1), in_config.dtype],
        x_sf: T.Tensor[x_sf_shape, in_config.sf_dtype],
        out: T.Tensor[(num_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[sf_shape, (out_sf_stride, 1), out_config.sf_dtype],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            x_ub = T.alloc_shared((block_m, block_k), in_config.dtype)
            out_ub = T.alloc_shared((block_m, block_k), out_config.dtype)
            versions = {x_ub: num_stages}
            if not sf_only:
                versions[out_ub] = num_stages

            if in_config.with_sf:
                dequant_ub = T.alloc_shared((block_m, block_k), value_dtype)
                if in_config.use_packed_ue8m0:
                    sf_in_ub = T.alloc_shared((align(num_groups, 32) // pack_factor,), T.uint16)
                else:
                    sf_in_ub = T.alloc_shared((align(num_groups, 8),), T.float32)
                versions[sf_in_ub] = num_stages
                versions[dequant_ub] = 1
            values_ub = dequant_ub if in_config.with_sf else x_ub

            sf_dense_ub = T.alloc_shared((block_m, sf_stride), T.uint8 if out_config.use_packed_ue8m0 else out_config.sf_dtype)
            if out_config.use_tma_aligned_col_major_sf:
                sf_out_ub = T.alloc_shared(
                    (num_groups // pack_factor, block_m * pack_factor) if out_config.use_packed_ue8m0 else (num_groups, block_m),
                    out_config.sf_dtype,
                )
                transpose_index_dtype = T.uint16 if out_config.use_packed_ue8m0 else T.uint32
                transpose_index_lanes = 128 if out_config.use_packed_ue8m0 else 64
                transpose_indices_ub = T.alloc_shared((transpose_index_lanes,), transpose_index_dtype)
            else:
                sf_out_ub = sf_dense_ub

            amax_ub = T.alloc_shared((block_m, sf_stride), T.float32)
            sf_inv_ub = T.alloc_shared((block_m, sf_stride), value_dtype)
            if not in_config.with_sf:
                if not cast_only:
                    versions[amax_ub] = num_stages
                versions[sf_inv_ub] = num_stages
                if out_config.use_tma_aligned_col_major_sf:
                    versions[sf_dense_ub] = num_stages
            T.annotate_buffer_versions(versions)

            if out_config.use_tma_aligned_col_major_sf:
                init_transpose_indices(transpose_indices_ub)

            for pid_token, pid_hidden in T.Persistent(
                [T.ceildiv(num_tokens, block_m), num_hidden_tiles],
                num_vec_cores,
                core_id,
                # Full last dimension: keeps the traversal row-major (token outer,
                # hidden tile inner). A smaller group size makes the realigned
                # PersistentFor walk the domain column-major, which costs DMA
                # locality on the wide shapes.
                group_size=num_hidden_tiles,
                num_stages=num_stages,
            ):
                col_offset = pid_hidden * block_k
                group_offset = pid_hidden * num_groups

                T.copy(x[pid_token * block_m, col_offset], x_ub)

                if in_config.with_sf:
                    input_sf_row = pid_token * block_m // in_config.sf_block[0]
                    if in_config.use_tma_aligned_col_major_sf:
                        if in_config.use_packed_ue8m0:
                            sf_in_matrix = T.Tensor((align(num_groups, 32) // pack_factor, pack_factor), T.uint8, sf_in_ub.data)
                            T.copy(
                                x_sf[group_offset // pack_factor, input_sf_row * pack_factor],
                                sf_in_matrix[0 : num_groups // pack_factor, 0:pack_factor],
                            )
                        else:
                            sf_in_matrix = T.Tensor((align(num_groups, 8), 1), T.float32, sf_in_ub.data)
                            T.copy(
                                x_sf[group_offset : group_offset + num_groups, input_sf_row : input_sf_row + 1],
                                sf_in_matrix[0:num_groups, 0:1],
                            )
                    else:
                        sf_in_flat = T.Tensor((align(num_groups, 32),), T.uint8, sf_in_ub.data) if in_config.use_packed_ue8m0 else sf_in_ub
                        T.copy(x_sf[input_sf_row, group_offset], sf_in_flat[0:num_groups])
                    dequantize_input(x_ub, sf_in_ub, dequant_ub)

                if cast_only:
                    T.copy(out_sf[pid_token * block_m, group_offset], sf_out_ub[0:block_m, 0:num_groups])

                if value_dtype == T.bfloat16:
                    quantize_bf16(values_ub, amax_ub, sf_inv_ub, sf_dense_ub, out_ub)
                else:
                    if stochastic_cast or (value_dtype == T.float32 and out_config.dtype == T.float8_e4m3fn and block_k >= 256):
                        step = 256
                    else:
                        step = 128
                    quantize_f32(values_ub, amax_ub, sf_inv_ub, sf_dense_ub, out_ub, pid_hidden, step)

                if out_config.use_tma_aligned_col_major_sf:
                    transpose_output_sf(sf_dense_ub, sf_out_ub, transpose_indices_ub)
                if not sf_only:
                    T.copy(out_ub, out[pid_token * block_m, col_offset])
                if not cast_only:
                    if out_config.use_tma_aligned_col_major_sf and out_config.use_packed_ue8m0:
                        T.copy(sf_out_ub, out_sf[group_offset // pack_factor, pid_token * block_m * pack_factor])
                    elif out_config.use_tma_aligned_col_major_sf:
                        T.copy(sf_out_ub, out_sf[group_offset, pid_token * block_m])
                    else:
                        T.copy(sf_out_ub[0:block_m, 0:num_groups], out_sf[pid_token * block_m, group_offset])

    return per_token_cast_kernel

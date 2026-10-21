import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from tile_kernels.quant.common import CastOutputConfig, get_packed_ue8m0_pack_factor, get_sf_shape
from tile_kernels.utils import align


@tilelang.jit
def get_swiglu_backward_kernel_asc(
    hidden: int,
    num_experts: int,
    alignment: int,
    with_weight: bool,
    with_routed_scaling: bool,
    use_clamp: bool,
    x_dtype: T.dtype,
    act_x_grad_dtype: T.dtype,
    out_config: CastOutputConfig,
    do_recompute: bool,
    num_vec_cores: int,
):
    num_stages = 2

    with_quant = out_config.with_sf
    out_dtype = T.bfloat16 if with_quant else out_config.dtype

    if with_quant:
        assert x_dtype == T.float8_e4m3fn
        assert out_config.dtype == T.float8_e4m3fn
        assert out_config.sf_block == (1, 32)
        assert not out_config.use_packed_ue8m0 or out_config.round_sf
    else:
        assert x_dtype in (T.bfloat16, T.float32)
        assert out_dtype in (T.bfloat16, T.float32)
    assert hidden % 128 == 0

    # Keep each SF column at the 32-byte DMA granularity.
    block_m = (16 if out_config.use_packed_ue8m0 else 8) if with_quant else 1
    if num_experts:
        assert alignment % block_m == 0, f'Ascend swiglu backward requires alignment % {block_m} == 0, got {alignment}'

    vec_size = 64
    block_k = min(hidden, 4096)
    while hidden % block_k != 0:
        block_k -= vec_size
    assert block_k >= 128 and hidden % block_k == 0

    # Padding maps only to unused 32-element amax groups.
    block_k_aligned = align(block_k, 256)

    pack_factor = get_packed_ue8m0_pack_factor() if with_quant and out_config.use_packed_ue8m0 else 1
    num_hidden_blocks = hidden // block_k
    num_groups = hidden // 32 if with_quant else 0
    num_block_groups = block_k // 32 if with_quant else 0
    num_stored_groups = num_groups // pack_factor
    num_stored_block_groups = num_block_groups // pack_factor
    # Pad each hidden block to one full vector store.
    sf_block_stride = align(num_stored_block_groups, vec_size // pack_factor) if with_quant else 0
    sf_half_stride = sf_block_stride * num_hidden_blocks
    sf_row_stride = sf_half_stride * 2
    if with_quant and out_config.use_tma_aligned_col_major_sf:
        sf_row_stride += 16 if out_config.use_packed_ue8m0 else 8
    num_scale_vectors = (num_block_groups + vec_size - 1) // vec_size if with_quant else 0
    sf_storage_dtype = T.uint16 if out_config.use_packed_ue8m0 else T.float32
    stored_groups_per_vector = vec_size * pack_factor // block_m
    # Pad transpose UB to avoid unsupported partial-vector PAT_VL masks.
    aligned_num_stored_block_groups = align(num_stored_block_groups, stored_groups_per_vector)

    num_expanded_tokens = T.dynamic('num_expanded_tokens')
    out_sf_stride = T.dynamic('out_sf_stride')
    x_sf_shape = (num_expanded_tokens, num_groups * 2) if with_quant else (1, 1)
    out_sf_shape = get_sf_shape((num_expanded_tokens, hidden * 2), out_config) if with_quant else (1, 1)
    x_grad_fp8_shape = (num_expanded_tokens, hidden * 2) if with_quant else (1, 1)

    @T.macro
    def vdiv_0ulp(a, b):
        """Correct a / b when the hardware quotient is finite; preserve signed zero."""
        z = S.vdiv(a, b, precision='ftz_true')
        y = S.vneg(b)
        r = S.alloc_var(T.float32)
        r = a
        S.vmula(r, z, y)
        z_pre = T.reinterpret(S.vadds(T.reinterpret(z, 'int32x64'), -1), 'float32x64')
        z_next = T.reinterpret(S.vadds(T.reinterpret(z, 'int32x64'), 1), 'float32x64')
        r_pre = S.alloc_var(T.float32)
        r_pre = a
        S.vmula(r_pre, z_pre, y)
        r_next = S.alloc_var(T.float32)
        r_next = a
        S.vmula(r_next, z_next, y)
        r = S.vabs(r)
        r_pre = S.vabs(r_pre)
        r_next = S.vabs(r_next)
        take_pre = S.vcmp(r_pre, r, op='le')
        best_r = S.vsel(r_pre, r, take_pre)
        best_z = S.vsel(z_pre, z, take_pre)
        take_next = S.vcmp(r_next, best_r, op='lt')
        corrected = S.vsel(z_next, best_z, take_next)
        is_zero = S.vcmps(z, 0.0, op='eq')
        return S.vsel(z, corrected, is_zero)

    @T.macro
    def sigmoid_vdiv_0ulp(a, b):
        """Correct a / b for sigmoid."""
        z = S.vdiv(a, b, precision='ftz_true')
        neg_b = S.vneg(b)

        residual = S.alloc_var(T.float32)
        residual = a
        S.vmula(residual, z, neg_b)

        residual_bits = T.reinterpret(residual, 'int32x64')
        sign_mask = S.vshrs(residual_bits, 31)
        use_bits_minus = S.vcmps(residual_bits, 0, op='lt')
        delta = S.vor(sign_mask, S.vdup(1, T.int32))
        adjacent = T.reinterpret(S.vadd(T.reinterpret(z, 'int32x64'), delta), 'float32x64')

        adjacent_residual = S.alloc_var(T.float32)
        adjacent_residual = a
        S.vmula(adjacent_residual, adjacent, neg_b)
        residual_sum = S.vadd(residual, adjacent_residual)
        take_bits_plus = S.vcmps(residual_sum, 0.0, op='gt')
        is_nonzero = S.vcmps(z, 0.0, op='ne')
        take_adjacent = S.pxor(take_bits_plus, use_bits_minus, is_nonzero)
        return S.vsel(adjacent, z, take_adjacent)

    @T.macro
    def get_scale_and_inverse(amax):
        # Compute one vector of output scales and inverse scales from per-group amax.
        clamped_amax = S.vmaxs(amax, out_config.clamp_min_value)
        if out_config.round_sf:
            exponent_mask = S.vdup(0x7F800000, T.uint32)
            exponent_carry = S.vdup(0x001FFFFF, T.uint32)
            inverse_bias = S.vdup(0x83000000, T.uint32)
            scale_bias = S.vdup(0x04000000, T.uint32)
            exponent = S.vand(S.vadd(T.reinterpret(clamped_amax, 'uint32x64'), exponent_carry), exponent_mask)
            scale = T.reinterpret(S.vsub(exponent, scale_bias), 'float32x64')
            inverse = T.reinterpret(S.vsub(inverse_bias, exponent), 'float32x64')
            return scale, inverse
        quant_max = S.vdup(448.0, T.float32)
        return S.vdiv(clamped_amax, quant_max), S.vdiv(quant_max, clamped_amax)

    @T.macro
    def store_fourfold_expanded(dst, offset, value):
        # Repeat each compact FP32 value four times for subsequent E2B_B32 loads.
        value_2_lo, value_2_hi = S.vintlv(value, value)
        value_4_0, value_4_1 = S.vintlv(value_2_lo, value_2_lo)
        value_4_2, value_4_3 = S.vintlv(value_2_hi, value_2_hi)
        S.vsts(dst[offset], value_4_0)
        S.vsts(dst[offset + vec_size], value_4_1)
        S.vsts(dst[offset + vec_size * 2], value_4_2)
        S.vsts(dst[offset + vec_size * 3], value_4_3)

    @T.macro
    def store_tile_amax(src, dst, tile):
        even01, odd01 = S.vld2(src[tile * 256], dist='DINTLV_B32')
        even23, odd23 = S.vld2(src[tile * 256 + 128], dist='DINTLV_B32')
        pair_max01 = S.vmax(S.vabs(even01), S.vabs(odd01))
        pair_max23 = S.vmax(S.vabs(even23), S.vabs(odd23))
        even, odd = S.vdintlv(pair_max01, pair_max23)
        amax = S.vcgmax(S.vmax(even, odd))
        store_mask = S.pset(32, 'PAT_VL8')
        S.vsts(dst[tile * 8], amax, store_mask, dist='NORM_B32', extent=8)

    @T.macro
    def init_transpose_indices(indices):
        # Precompute gather indices shared by every token block when converting
        # dense token-major scales to the output layout.
        with T.SimdVF():
            if out_config.use_packed_ue8m0:
                lane = S.vci(0, T.int16)
                row = S.vand(T.reinterpret(lane, 'uint16x128'), S.vdup(block_m - 1, T.uint16))
                group = T.reinterpret(S.vshrs(lane, block_m.bit_length() - 1), 'uint16x128')
                source = S.vadd(S.vmuls(row, sf_row_stride), group)
                S.vsts(indices[0], source, dist='NORM_B16')
            else:
                lane = S.vci(0, T.int32)
                row = S.vand(T.reinterpret(lane, 'uint32x64'), S.vdup(block_m - 1, T.uint32))
                group = T.reinterpret(S.vshrs(lane, block_m.bit_length() - 1), 'uint32x64')
                source = S.vadd(S.vmuls(row, sf_row_stride), group)
                S.vsts(indices[0], source, dist='NORM_B32')

    @T.macro
    def transpose_sf(src, dst, indices, source_col):
        # Transpose one hidden block of dense scales into the stored layout.
        with T.SimdVF():
            source_indices = S.vld(indices[0])
            store_dist = 'NORM_B16' if out_config.use_packed_ue8m0 else 'NORM_B32'
            for output_vector in T.unroll((num_stored_block_groups + stored_groups_per_vector - 1) // stored_groups_per_vector, explicit=True):
                group_base = output_vector * stored_groups_per_vector
                values = S.vgather2(src[0, source_col + group_base], source_indices)
                S.vsts(dst[group_base, 0], values, dist=store_dist)

    @T.prim_func
    def swiglu_backward_kernel_asc(
        x: T.Tensor[(num_expanded_tokens, hidden * 2), x_dtype],
        x_sf: T.Tensor[x_sf_shape, T.float32],
        act_x_grad: T.Tensor[(num_expanded_tokens, hidden), act_x_grad_dtype],
        topk_weights: T.Tensor[(num_expanded_tokens,), T.float32],
        routed_scaling_factor: T.float32,
        psum_num_tokens_per_expert: T.Tensor[(num_experts,), T.int32],
        out: T.Tensor[(num_expanded_tokens, hidden), out_dtype],
        x_grad_fp8: T.Tensor[x_grad_fp8_shape, out_config.dtype],
        x_grad_fp8_sf: T.StridedTensor[out_sf_shape, (out_sf_stride, 1), out_config.sf_dtype],
        x_grad: T.Tensor[(num_expanded_tokens, hidden * 2), out_dtype],
        weight_grad: T.Tensor[(num_expanded_tokens,), T.float32],
        clamp_value: T.float32,
    ):
        with T.Kernel(num_vec_cores) as core_id:
            x_ub = T.alloc_shared((block_k,), x_dtype)
            y_ub = T.alloc_shared((block_k,), x_dtype)
            grad_ub = T.alloc_shared((block_k,), act_x_grad_dtype)
            x_grad_ub = T.alloc_shared((block_k,), out_dtype)
            y_grad_ub = T.alloc_shared((block_k,), out_dtype)
            if with_quant:
                x_sf_ub = T.alloc_shared((sf_block_stride * pack_factor,), T.float32)
                y_sf_ub = T.alloc_shared((sf_block_stride * pack_factor,), T.float32)
                x_grad_f32_ub = T.alloc_shared((block_k_aligned,), T.float32)
                y_grad_f32_ub = T.alloc_shared((block_k_aligned,), T.float32)
                x_grad_fp8_ub = T.alloc_shared((block_k,), T.float8_e4m3fn)
                y_grad_fp8_ub = T.alloc_shared((block_k,), T.float8_e4m3fn)
                # Reused for expanded input scales, amax, and inverse scales.
                x_quant_scratch_ub = T.alloc_shared((sf_block_stride * pack_factor * 4,), T.float32)
                y_quant_scratch_ub = T.alloc_shared((sf_block_stride * pack_factor * 4,), T.float32)
                # Packed SF is written as bytes and transposed through a uint16 view.
                sf_dense_ub = T.alloc_shared((block_m, sf_row_stride), sf_storage_dtype)
                if out_config.use_packed_ue8m0:
                    sf_dense_u8 = T.Tensor((block_m, sf_row_stride * 2), T.uint8, sf_dense_ub.data)
                if out_config.use_tma_aligned_col_major_sf:
                    x_sf_out_ub = T.alloc_shared((aligned_num_stored_block_groups, block_m), sf_storage_dtype)
                    y_sf_out_ub = T.alloc_shared((aligned_num_stored_block_groups, block_m), sf_storage_dtype)
                    transpose_indices_ub = T.alloc_shared((vec_size * pack_factor,), T.uint16 if out_config.use_packed_ue8m0 else T.uint32)

            if with_weight:
                weight_acc_ub = T.alloc_shared((block_m, vec_size), T.float32)
                weight_out_ub = T.alloc_shared((vec_size,), T.float32)
            if do_recompute:
                act_out_ub = T.alloc_shared((block_k,), out_dtype)
            if num_experts:
                psum_ub = T.alloc_shared((num_experts,), T.int32)

            versions = {
                x_ub: num_stages,
                y_ub: num_stages,
                grad_ub: num_stages,
                x_grad_ub: num_stages,
                y_grad_ub: num_stages,
            }
            if with_quant:
                versions.update(
                    {
                        x_sf_ub: num_stages,
                        y_sf_ub: num_stages,
                        # Vector-local scratch does not need MTE overlap versions.
                        x_grad_f32_ub: 1,
                        y_grad_f32_ub: 1,
                        x_grad_fp8_ub: num_stages,
                        y_grad_fp8_ub: num_stages,
                        x_quant_scratch_ub: 1,
                        y_quant_scratch_ub: 1,
                        sf_dense_ub: 1,
                    }
                )
            if with_quant and out_config.use_tma_aligned_col_major_sf:
                versions[x_sf_out_ub] = 1
                versions[y_sf_out_ub] = 1
                versions[transpose_indices_ub] = 1
            if with_weight:
                versions[weight_acc_ub] = 1
                versions[weight_out_ub] = num_stages
            if do_recompute:
                versions[act_out_ub] = num_stages
            if num_experts:
                versions[psum_ub] = 1
            T.annotate_buffer_versions(versions)

            if with_quant and out_config.use_tma_aligned_col_major_sf:
                init_transpose_indices(transpose_indices_ub)
            if num_experts:
                T.copy(psum_num_tokens_per_expert, psum_ub[:num_experts])
                expert_id = T.alloc_var(T.int32, init=0)
                valid_token_end = T.alloc_var(T.int32, init=0)

            num_token_blocks = num_expanded_tokens // block_m
            blocks_per_core = num_token_blocks // num_vec_cores
            num_long_cores = num_token_blocks % num_vec_cores
            # Assign contiguous block ranges per core to spread concurrent GM traffic across L2 home paths.
            core_block_start = core_id * blocks_per_core + T.min(core_id, num_long_cores)

            for scheduler_pid in T.Persistent(
                [num_token_blocks],
                num_vec_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
            ):
                wave = scheduler_pid // num_vec_cores
                token_block = core_block_start + wave
                token_base = token_block * block_m
                if num_experts:
                    while expert_id < num_experts and token_base >= align(psum_ub[expert_id], alignment):
                        expert_id += 1
                    if expert_id < num_experts:
                        valid_token_end = psum_ub[expert_id]
                    else:
                        valid_token_end = 0

                for hidden_block in range(num_hidden_blocks):
                    col_offset = hidden_block * block_k
                    sf_offset = hidden_block * num_block_groups
                    sf_block_offset = hidden_block * sf_block_stride

                    for row in T.serial(block_m):
                        token_id = token_base + row
                        valid_token_f32 = T.alloc_var(T.float32, init=1.0)
                        if num_experts:
                            valid_token_f32 = T.cast(expert_id < num_experts and token_id < valid_token_end, T.float32)

                        effective_weight = T.alloc_var(T.float32, init=valid_token_f32)
                        if with_weight:
                            effective_weight = topk_weights[token_id] * valid_token_f32
                            if with_routed_scaling:
                                effective_weight = effective_weight * routed_scaling_factor

                        T.copy(x[token_id, col_offset : col_offset + block_k], x_ub)
                        T.copy(x[token_id, hidden + col_offset : hidden + col_offset + block_k], y_ub)
                        if with_quant:
                            T.copy(x_sf[token_id, sf_offset : sf_offset + num_block_groups], x_sf_ub[:num_block_groups])
                            T.copy(x_sf[token_id, num_groups + sf_offset : num_groups + sf_offset + num_block_groups], y_sf_ub[:num_block_groups])
                        T.copy(act_x_grad[token_id, col_offset : col_offset + block_k], grad_ub)

                        with T.SimdVF():
                            if with_quant:
                                for scale_vector in range(num_scale_vectors):
                                    scale_offset = scale_vector * vec_size
                                    expanded_offset = scale_offset * 4
                                    x_scale_compact = S.vld(x_sf_ub[scale_offset])
                                    y_scale_compact = S.vld(y_sf_ub[scale_offset])
                                    store_fourfold_expanded(x_quant_scratch_ub, expanded_offset, x_scale_compact)
                                    store_fourfold_expanded(y_quant_scratch_ub, expanded_offset, y_scale_compact)
                                S.mem_bar('VST_VLD')
                            ones = S.vdup(1.0, T.float32)
                            zeros = S.vdup(0.0, T.float32)
                            if with_weight:
                                weight_acc = S.alloc_var(T.float32)
                                weight_acc = zeros if hidden_block == 0 else S.vld(weight_acc_ub[row, 0])

                            for vector in T.serial(block_k // vec_size):
                                col = vector * vec_size
                                if with_quant:
                                    x_scale = S.vld(x_quant_scratch_ub[vector * 8], dist='E2B_B32')
                                    y_scale = S.vld(y_quant_scratch_ub[vector * 8], dist='E2B_B32')
                                    x_raw = S.vcvt(S.vld(x_ub[col], dist='UNPK4_B8'), T.float32)
                                    y_raw = S.vcvt(S.vld(y_ub[col], dist='UNPK4_B8'), T.float32)
                                    x_value = S.vmul(x_raw, x_scale)
                                    y_value = S.vmul(y_raw, y_scale)
                                elif x_dtype == T.bfloat16:
                                    x_value = S.vcvt(S.vld(x_ub[col], dist='UNPK_B16'), T.float32, part=0)
                                    y_value = S.vcvt(S.vld(y_ub[col], dist='UNPK_B16'), T.float32, part=0)
                                else:
                                    x_value = S.vld(x_ub[col])
                                    y_value = S.vld(y_ub[col])
                                if act_x_grad_dtype == T.bfloat16:
                                    grad = S.vcvt(S.vld(grad_ub[col], dist='UNPK_B16'), T.float32, part=0)
                                else:
                                    grad = S.vld(grad_ub[col])

                                if use_clamp:
                                    is_clamped_x = S.vcmps(x_value, clamp_value, op='gt')
                                    is_clamped_y = S.vcmps(S.vabs(y_value), clamp_value, op='gt')
                                    x_compute_value = S.vmins(x_value, clamp_value)
                                    y_compute_value = S.vmins(S.vmaxs(y_value, -clamp_value), clamp_value)
                                else:
                                    x_compute_value = x_value
                                    y_compute_value = y_value

                                sigmoid_denominator = S.vadds(S.vexpdif(zeros, x_compute_value), 1.0)
                                sigmoid = sigmoid_vdiv_0ulp(ones, sigmoid_denominator)
                                if with_weight or do_recompute:
                                    # Keep a separate exact division for the required FP32 rounding.
                                    act_value = S.vmul(vdiv_0ulp(x_compute_value, sigmoid_denominator), y_compute_value)
                                if with_weight:
                                    weight_contribution = S.vmul(grad, act_value)
                                    weight_value = S.vmuls(weight_contribution, valid_token_f32) if num_experts else weight_contribution
                                    weight_acc = S.vadd(weight_acc, weight_value)
                                grad_ws = S.vmul(S.vmuls(grad, effective_weight), sigmoid)
                                inner = S.vadd(ones, S.vmul(x_compute_value, S.vsub(ones, sigmoid)))
                                x_grad_unclamped = S.vmul(S.vmul(grad_ws, y_compute_value), inner)
                                y_grad_unclamped = S.vmul(grad_ws, x_compute_value)
                                x_grad_value = S.vsel(zeros, x_grad_unclamped, is_clamped_x) if use_clamp else x_grad_unclamped
                                y_grad_value = S.vsel(zeros, y_grad_unclamped, is_clamped_y) if use_clamp else y_grad_unclamped
                                if not with_quant:
                                    if out_dtype == T.bfloat16:
                                        S.vsts(x_grad_ub[col], S.vcvt(x_grad_value, T.bfloat16), dist='PK_B32')
                                        S.vsts(y_grad_ub[col], S.vcvt(y_grad_value, T.bfloat16), dist='PK_B32')
                                    else:
                                        S.vsts(x_grad_ub[col], x_grad_value)
                                        S.vsts(y_grad_ub[col], y_grad_value)
                                else:
                                    S.vsts(x_grad_f32_ub[col], x_grad_value)
                                    S.vsts(y_grad_f32_ub[col], y_grad_value)
                                if do_recompute:
                                    act_out_value = S.vmuls(act_value, effective_weight)
                                    if out_dtype == T.bfloat16:
                                        S.vsts(act_out_ub[col], S.vcvt(act_out_value, T.bfloat16), dist='PK_B32')
                                    else:
                                        S.vsts(act_out_ub[col], act_out_value)

                            if with_weight:
                                one_mask = S.pset(32, 'PAT_VL1')
                                if hidden_block + 1 == num_hidden_blocks:
                                    reduced_weight_sum = S.vcadd(weight_acc)
                                    weight_sum = S.vmuls(reduced_weight_sum, routed_scaling_factor) if with_routed_scaling else reduced_weight_sum
                                    S.vsts(weight_out_ub[0], weight_sum, one_mask, dist='ONEPT_B32')
                                else:
                                    S.vsts(weight_acc_ub[row, 0], weight_acc)

                            if with_quant:
                                S.mem_bar('VST_VLD')
                                # Batch amax independently to keep the derivative loop compact.
                                for tile in T.serial(block_k_aligned // 256):
                                    store_tile_amax(x_grad_f32_ub, x_quant_scratch_ub, tile)
                                    store_tile_amax(y_grad_f32_ub, y_quant_scratch_ub, tile)
                                S.mem_bar('VST_VLD')
                                if num_experts:
                                    valid_token_vec = S.vdup(valid_token_f32, T.float32)
                                # Preserve unread amax while overwriting the scratch with inverses.
                                for scale_vector_index in range(num_scale_vectors):
                                    scale_vector = num_scale_vectors - 1 - scale_vector_index
                                    scale_offset = scale_vector * vec_size
                                    inverse_offset = scale_offset * 4
                                    x_scale_value, x_inv = get_scale_and_inverse(S.vld(x_quant_scratch_ub[scale_offset]))
                                    y_scale_value, y_inv = get_scale_and_inverse(S.vld(y_quant_scratch_ub[scale_offset]))
                                    store_fourfold_expanded(x_quant_scratch_ub, inverse_offset, x_inv)
                                    store_fourfold_expanded(y_quant_scratch_ub, inverse_offset, y_inv)
                                    x_scale_out = S.vmul(x_scale_value, valid_token_vec) if num_experts else x_scale_value
                                    y_scale_out = S.vmul(y_scale_value, valid_token_vec) if num_experts else y_scale_value
                                    if out_config.use_packed_ue8m0:
                                        S.vsts(
                                            sf_dense_u8[row, sf_block_offset * 2 + scale_offset],
                                            T.reinterpret(S.vshrs(T.reinterpret(x_scale_out, 'uint32x64'), 23), 'uint8x256'),
                                            dist='PK4_B32',
                                        )
                                        S.vsts(
                                            sf_dense_u8[row, sf_half_stride * 2 + sf_block_offset * 2 + scale_offset],
                                            T.reinterpret(S.vshrs(T.reinterpret(y_scale_out, 'uint32x64'), 23), 'uint8x256'),
                                            dist='PK4_B32',
                                        )
                                    else:
                                        S.vsts(sf_dense_ub[row, sf_block_offset + scale_offset], x_scale_out)
                                        S.vsts(sf_dense_ub[row, sf_half_stride + sf_block_offset + scale_offset], y_scale_out)

                                S.mem_bar('VST_VLD')
                                if num_experts:
                                    valid_token_mask = S.vcmps(valid_token_vec, 0.0, op='gt')
                                for vector in T.serial(block_k // vec_size):
                                    col = vector * vec_size
                                    x_inv_vec = S.vld(x_quant_scratch_ub[vector * 8], dist='E2B_B32')
                                    y_inv_vec = S.vld(y_quant_scratch_ub[vector * 8], dist='E2B_B32')
                                    x_grad_quant_value = S.vld(x_grad_f32_ub[col])
                                    y_grad_quant_value = S.vld(y_grad_f32_ub[col])
                                    x_scaled_value = S.vmul(x_grad_quant_value, x_inv_vec)
                                    y_scaled_value = S.vmul(y_grad_quant_value, y_inv_vec)
                                    S.vsts(x_grad_ub[col], S.vcvt(x_grad_quant_value, T.bfloat16), dist='PK_B32')
                                    S.vsts(y_grad_ub[col], S.vcvt(y_grad_quant_value, T.bfloat16), dist='PK_B32')
                                    x_scaled = S.vsel(x_scaled_value, zeros, valid_token_mask) if num_experts else x_scaled_value
                                    y_scaled = S.vsel(y_scaled_value, zeros, valid_token_mask) if num_experts else y_scaled_value
                                    S.vsts(x_grad_fp8_ub[col], S.vcvt(x_scaled, T.float8_e4m3fn), dist='PK4_B32')
                                    S.vsts(y_grad_fp8_ub[col], S.vcvt(y_scaled, T.float8_e4m3fn), dist='PK4_B32')

                        T.copy(x_grad_ub, x_grad[token_id, col_offset : col_offset + block_k])
                        T.copy(y_grad_ub, x_grad[token_id, hidden + col_offset : hidden + col_offset + block_k])
                        if with_quant:
                            T.copy(x_grad_fp8_ub, x_grad_fp8[token_id, col_offset : col_offset + block_k])
                            T.copy(y_grad_fp8_ub, x_grad_fp8[token_id, hidden + col_offset : hidden + col_offset + block_k])
                        if do_recompute:
                            T.copy(act_out_ub, out[token_id, col_offset : col_offset + block_k])
                        if with_weight and hidden_block + 1 == num_hidden_blocks:
                            T.copy(weight_out_ub[:1], weight_grad[token_id : token_id + 1])

                if with_quant:
                    if out_config.use_packed_ue8m0:
                        if out_config.use_tma_aligned_col_major_sf:
                            output_sf = T.Tensor((num_stored_groups * 2, num_expanded_tokens), T.uint16, x_grad_fp8_sf.data)
                        else:
                            output_sf = T.Tensor((num_expanded_tokens, num_stored_groups * 2), T.uint16, x_grad_fp8_sf.data)
                    else:
                        output_sf = x_grad_fp8_sf

                    for output_block in range(num_hidden_blocks):
                        sf_source_offset = output_block * sf_block_stride
                        sf_output_offset = output_block * num_stored_block_groups
                        if out_config.use_tma_aligned_col_major_sf:
                            transpose_sf(sf_dense_ub, x_sf_out_ub, transpose_indices_ub, sf_source_offset)
                            T.copy(x_sf_out_ub[0:num_stored_block_groups, 0:block_m], output_sf[sf_output_offset, token_base])
                            transpose_sf(sf_dense_ub, y_sf_out_ub, transpose_indices_ub, sf_half_stride + sf_source_offset)
                            T.copy(y_sf_out_ub[0:num_stored_block_groups, 0:block_m], output_sf[num_stored_groups + sf_output_offset, token_base])
                        else:
                            T.copy(
                                sf_dense_ub[0:block_m, sf_source_offset : sf_source_offset + num_stored_block_groups],
                                output_sf[token_base, sf_output_offset],
                            )
                            T.copy(
                                sf_dense_ub[
                                    0:block_m,
                                    sf_half_stride + sf_source_offset : sf_half_stride + sf_source_offset + num_stored_block_groups,
                                ],
                                output_sf[token_base, num_stored_groups + sf_output_offset],
                            )

    return swiglu_backward_kernel_asc

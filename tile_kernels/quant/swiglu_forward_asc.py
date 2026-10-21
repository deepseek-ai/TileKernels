import math

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from tile_kernels.quant.common import CastOutputConfig, get_packed_ue8m0_pack_factor, get_sf_shape
from tile_kernels.utils import align


@tilelang.jit
def get_swiglu_forward_kernel_asc(
    hidden: int,
    num_experts: int,
    alignment: int,
    with_weight: bool,
    with_routed_scaling: bool,
    clamp_value,
    count_clamp: bool,
    in_dtype: T.dtype,
    out_config: CastOutputConfig,
    num_vec_cores: int,
):
    has_clamp = clamp_value is not None
    has_psum = num_experts > 0

    assert not count_clamp or has_clamp
    assert not with_routed_scaling or with_weight
    assert hidden % 128 == 0, f'Ascend swiglu requires hidden % 128 == 0, got {hidden}'
    assert not out_config.use_tma_aligned_col_major_sf or out_config.use_packed_ue8m0
    if out_config.with_sf:
        assert out_config.sf_block == (1, 32), 'Ascend swiglu only supports sf_block=(1, 32)'

    token_group = 32 if out_config.with_sf and out_config.use_tma_aligned_col_major_sf else 1
    if has_psum:
        assert alignment % token_group == 0, f'Ascend swiglu col-major path requires alignment % {token_group} == 0, got {alignment}'
    use_compact_groups = has_psum and token_group > 1
    # Fast certificate is only enabled for bf16 -> e4m3 with power-of-two sf.
    # It checks that fast and exact division give the same e4m3/sf result;
    # unsafe tokens fall back to exact division.
    use_fast_certificate = in_dtype == T.bfloat16 and out_config.with_sf and out_config.round_sf and out_config.use_tma_aligned_col_major_sf
    num_stages = 3
    # Multiplication by a weight requires a larger certificate error bound.
    certificate_error = 5 if with_weight else 2
    vec_size = 64
    num_vecs = hidden // vec_size
    hidden_aligned = align(hidden, 256)

    num_expanded_tokens = T.dynamic('num_expanded_tokens')
    sf_stride = T.dynamic('sf_stride')
    sf_shape = get_sf_shape((num_expanded_tokens, hidden), out_config) if out_config.with_sf else (1, 1)
    num_groups = hidden // 32
    pack_factor = get_packed_ue8m0_pack_factor()

    @T.macro
    def get_scale_and_inverse(amax):
        clamped_amax = S.vmaxs(amax, out_config.clamp_min_value)
        scale_bits = T.reinterpret(S.vmuls(clamped_amax, 1.0 / 448.0), 'uint32x64')
        exponent = S.vadds(S.vshrs(S.vsub(scale_bits, S.vdup(1, T.uint32)), 23), 1)
        inverse = T.reinterpret(S.vshls(S.vsub(S.vdup(254, T.uint32), exponent), 23), 'float32x64')
        if out_config.use_packed_ue8m0:
            return exponent, inverse
        return T.reinterpret(S.vshls(exponent, 23), 'float32x64'), inverse

    @T.macro
    def init_transpose_indices(indices):
        # Precompute gather indices shared by every token group when converting the
        # dense token-major scale buffer to the packed col-major output layout.
        with T.SimdVF():
            lane = S.vci(0, T.int16)
            output_row = S.vand(T.reinterpret(lane, 'uint16x128'), S.vdup(token_group - 1, T.uint16))
            group_in_vector = T.reinterpret(S.vshrs(lane, int(math.log2(token_group))), 'uint16x128')
            source_indices = S.vadd(S.vmuls(output_row, align(num_groups, 64) // pack_factor), group_in_vector)
            S.vsts(indices[0], source_indices, dist='NORM_B16', extent=128)

    @T.macro
    def transpose_packed_scales(src, dst, indices):
        # Transpose one token group's packed UE8M0 scales.
        with T.SimdVF():
            groups_per_vector = 128 // token_group
            num_output_values = num_groups // pack_factor * token_group
            num_full_vectors = num_output_values // 128
            num_tail_values = num_output_values % 128
            source_indices = S.vld(indices[0])
            src_u16 = T.Tensor((token_group, align(num_groups, 64) // pack_factor), T.uint16, src.data)

            if num_full_vectors > 0:
                for output_vector in T.unroll(num_full_vectors, explicit=True):
                    group_base = output_vector * groups_per_vector
                    values = S.vgather2(src_u16[0, group_base], source_indices)
                    S.vsts(dst[group_base, 0], values, dist='NORM_B16', extent=128)

            if num_tail_values > 0:
                tail_group_base = num_full_vectors * groups_per_vector
                tail_mask = S.pset(16, f'PAT_VL{num_tail_values}')
                values = S.vgather2(src_u16[0, tail_group_base], source_indices, tail_mask)
                S.vsts(dst[tail_group_base, 0], values, tail_mask, dist='NORM_B16', extent=num_tail_values)

    @T.macro
    def compute_swiglu(xl_ub, xr_ub, values_ub, out_ub, weight, precision, amax_ub, with_certificate, pacc_buf, count_clamp, acc=None):
        # Compute SiLU(x_l) * x_r, optionally collecting clamp counters,
        # quantization amax values, and the fast-division safety certificate.
        zeros = S.vdup(0.0, T.float32)
        if count_clamp:
            if acc is None:
                acc = S.alloc_local((3,), T.int32)
                for i in T.unroll(3, explicit=True):
                    acc[i] = S.vdup(0, T.int32)
        if with_certificate:
            certificate_min = S.alloc_var(T.uint32)
            certificate_max = S.alloc_var(T.uint32)
            certificate_min = S.vdup(0xFFFFFFFF, T.uint32)
            certificate_max = S.vdup(0, T.uint32)
        else:
            certificate_min = None
            certificate_max = None
        if out_config.with_sf:
            amax_store_mask = S.pset(32, 'PAT_VL8')
            amax_acc = S.alloc_local((4,), T.float32)
            for i in T.unroll(4, explicit=True):
                amax_acc[i] = S.vdup(0.0, T.float32)

        for tile_id in T.serial(T.ceildiv(num_vecs, 4)):
            for sub in T.unroll(4, explicit=True):
                vector_id = tile_id * 4 + sub
                if vector_id < num_vecs:
                    col = vector_id * vec_size
                    if in_dtype == T.bfloat16:
                        val_l = S.vcvt(S.vld(xl_ub[col], dist='UNPK_B16'), T.float32, part=0)
                        val_r = S.vcvt(S.vld(xr_ub[col], dist='UNPK_B16'), T.float32, part=0)
                    else:
                        val_l = S.vld(xl_ub[col])
                        val_r = S.vld(xr_ub[col])
                    if has_clamp:
                        silu_l = S.vmins(val_l, clamp_value)
                        mul_r = S.vmaxs(S.vmins(val_r, clamp_value), -clamp_value)
                    else:
                        silu_l = val_l
                        mul_r = val_r
                    denominator = S.vadds(S.vexpdif(zeros, silu_l), 1.0)
                    quotient = S.vdiv(silu_l, denominator, precision=precision)
                    val = S.vmul(quotient, mul_r)
                    out_val = S.vmuls(val, weight) if with_weight else val

                    if out_config.with_sf:
                        S.vsts(values_ub[col], out_val)
                    elif out_config.dtype == T.bfloat16:
                        S.vsts(out_ub[col], S.vcvt(out_val, T.bfloat16), dist='PK_B32')
                    else:
                        S.vsts(out_ub[col], out_val)

                    if with_certificate:
                        # Keep the low 19 mantissa bits; values too close to either end of this window use the exact fallback.
                        metric = S.vand(T.reinterpret(out_val, 'uint32x64'), S.vdup(0x7FFFF, T.uint32))
                        certificate_min = S.vmin(certificate_min, metric)
                        certificate_max = S.vmax(certificate_max, metric)
                    if count_clamp:
                        clamped_mask_0 = S.vcmps(val_l, clamp_value, op='gt')
                        clamped_mask_1 = S.vcmps(val_r, clamp_value, op='gt')
                        clamped_mask_2 = S.vcmps(val_r, -clamp_value, op='lt')
                        acc[0] = S.vadds(acc[0], 1, clamped_mask_0, mode='MODE_MERGING')
                        acc[1] = S.vadds(acc[1], 1, clamped_mask_1, mode='MODE_MERGING')
                        acc[2] = S.vadds(acc[2], 1, clamped_mask_2, mode='MODE_MERGING')
                    if out_config.with_sf:
                        amax_acc[sub] = S.vabs(out_val)
            if out_config.with_sf:
                # 256-element tile amax reduction
                low01, high01 = S.vdintlv(amax_acc[0], amax_acc[1])
                low23, high23 = S.vdintlv(amax_acc[2], amax_acc[3])
                compressed01 = S.vmax(low01, high01)
                compressed23 = S.vmax(low23, high23)
                low, high = S.vdintlv(compressed01, compressed23)
                amax = S.vcgmax(S.vmax(low, high))
                S.vsts(amax_ub[tile_id * 8], amax, amax_store_mask, dist='NORM_B32', extent=8)
        if count_clamp:
            for i in T.unroll(3, explicit=True):
                S.vsts(pacc_buf[i, 0], S.vadd(S.vld(pacc_buf[i, 0]), acc[i]))
        return certificate_min, certificate_max

    @T.macro
    def quantize_f32(values_ub, amax_ub, inverse_ub, sf_dense_ub, out_ub, token_offset):
        # Get inverse scale and quantize FP32 SwiGLU result to FP8.
        with T.SimdVF():
            for group_batch in T.serial(T.ceildiv(num_groups, 64)):
                group = group_batch * 64
                amax = S.vld(amax_ub[group])
                if out_config.round_sf:
                    scales_or_exponents, inverses = get_scale_and_inverse(amax)
                    if out_config.use_packed_ue8m0:
                        S.vsts(sf_dense_ub[token_offset, group], T.reinterpret(scales_or_exponents, 'uint8x256'), dist='PK4_B32')
                    else:
                        S.vsts(sf_dense_ub[token_offset, group], scales_or_exponents)
                    S.vsts(inverse_ub[group], inverses)
                else:
                    clamped_amax = S.vmaxs(amax, out_config.clamp_min_value)
                    quant_max = S.vdup(448.0, T.float32)
                    S.vsts(sf_dense_ub[token_offset, group], S.vdiv(clamped_amax, quant_max))
                    S.vsts(inverse_ub[group], S.vdiv(quant_max, clamped_amax))

            S.mem_bar('VST_VLD')
            group_lane = S.vshrs(S.vci(0, T.int32), 5)
            for group_batch in T.unroll(T.ceildiv(num_groups, 64), explicit=True):
                group_base = group_batch * 64
                vector_base = group_batch * 32
                inverses = S.vld(inverse_ub[group_base])
                for vector in T.serial(T.min(32, num_vecs - vector_base)):
                    col = (vector_base + vector) * vec_size
                    inverse_indices = S.vadds(group_lane, vector * 2)
                    expanded_inverse = S.vselr(inverses, inverse_indices)
                    quantized = S.vcvt(S.vmul(S.vld(values_ub[col]), expanded_inverse), T.float8_e4m3fn)
                    S.vsts(out_ub[col], quantized, dist='PK4_B32')

    @T.macro
    def load_weight(topk_weights, token_id, routed_scaling_factor):
        weight = T.alloc_var(T.float32, init=1.0)
        if with_weight:
            weight = topk_weights[token_id]
            if with_routed_scaling:
                weight = weight * routed_scaling_factor
        return weight

    @T.macro
    def is_valid_token(token_id, expert_id, psum_ub):
        valid = T.alloc_var(T.bool, init=token_id < num_expanded_tokens)
        if has_psum:
            valid = valid and (expert_id < num_experts) and (token_id < psum_ub[expert_id])
        return valid

    @T.macro
    def store_sf_row(sf_dense_ub, out_sf, row, token_id):
        if out_config.with_sf and not out_config.use_tma_aligned_col_major_sf:
            T.copy(sf_dense_ub[row, 0:num_groups], out_sf[token_id, 0])

    @T.macro
    def load_inputs(x_gm, xl, xr, topk_w, rsf, tid):
        weight = load_weight(topk_w, tid, rsf)
        T.copy(x_gm[tid, 0], xl)
        T.copy(x_gm[tid, hidden], xr)
        return weight

    @T.macro
    def finish_token(vals, out, amax, inv, sf, out_gm, out_sf_gm, off, tid, count_buf, count_clamp):
        if out_config.with_sf:
            quantize_f32(vals, amax, inv, sf, out, off)
        T.copy(out, out_gm[tid, 0])
        store_sf_row(sf, out_sf_gm, off, tid)
        if count_clamp:
            count_buf[0] += hidden

    @T.prim_func
    def swiglu_forward_kernel(
        x: T.Tensor[(num_expanded_tokens, hidden * 2), in_dtype],
        out: T.Tensor[(num_expanded_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[sf_shape, (sf_stride, 1), out_config.sf_dtype],
        psum_num_tokens_per_expert: T.Tensor[num_experts, T.int32],
        topk_weights: T.Tensor[(num_expanded_tokens,), T.float32],
        routed_scaling_factor: T.float32,
        clamped_count: T.Tensor[(4,), T.int64],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            if count_clamp:
                pacc_ub = T.alloc_shared((3, vec_size), T.int32)  # Per-lane counts; reduce in int64 at the end.
                count_ub = T.alloc_shared((1,), T.int64)

                count_ub[0] = 0
                with T.SimdVF():
                    T.clear(pacc_ub)
            else:
                count_ub = None
                pacc_ub = None

            if has_psum:
                psum_ub = T.alloc_shared((num_experts,), T.int32)
                T.copy(psum_num_tokens_per_expert, psum_ub)
                expert_id = T.alloc_var(T.int32, init=0)
                if use_compact_groups:
                    group_psum_ub = T.alloc_shared((num_experts,), T.int32)
                    group_psum_ub[0] = T.ceildiv(psum_ub[0], token_group)
                    for expert in T.serial(1, num_experts):
                        expert_start = (psum_ub[expert - 1] + alignment - 1) // alignment * alignment
                        expert_groups = T.ceildiv(psum_ub[expert] - expert_start, token_group)
                        group_psum_ub[expert] = group_psum_ub[expert - 1] + expert_groups

            xl_ub = T.alloc_shared((hidden_aligned,), in_dtype)
            xr_ub = T.alloc_shared((hidden_aligned,), in_dtype)
            out_ub = T.alloc_shared((hidden_aligned,), out_config.dtype)
            values_ub = T.alloc_shared((hidden_aligned,), T.float32)
            if use_fast_certificate:
                xl_fb_ub = T.alloc_shared((hidden_aligned,), in_dtype)
                xr_fb_ub = T.alloc_shared((hidden_aligned,), in_dtype)
                out_fb_ub = T.alloc_shared((hidden_aligned,), out_config.dtype)
                values_fb_ub = T.alloc_shared((hidden_aligned,), T.float32)
            if out_config.with_sf:
                amax_ub = T.alloc_shared((align(num_groups, 64),), T.float32)
                if use_fast_certificate:
                    certificate_ub = T.alloc_shared((token_group, 2), T.uint32)
                inverse_ub = T.alloc_shared((align(num_groups, 64),), T.float32)
                sf_dense_ub = T.alloc_shared(
                    (token_group, align(num_groups, 64)),
                    T.uint8 if out_config.use_packed_ue8m0 else T.float32,
                )
                if out_config.use_tma_aligned_col_major_sf:
                    sf_ub = T.alloc_shared((num_groups // pack_factor, token_group), T.uint16)
                    transpose_indices_ub = T.alloc_shared((128,), T.uint16)
            else:
                amax_ub = None
                inverse_ub = None
                sf_dense_ub = None
            T.annotate_buffer_versions({xl_ub: num_stages, xr_ub: num_stages, out_ub: num_stages})

            if out_config.with_sf and out_config.use_tma_aligned_col_major_sf:
                init_transpose_indices(transpose_indices_ub)

            for pid in T.Persistent(
                [T.ceildiv(num_expanded_tokens, token_group)],
                num_vec_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
            ):
                token_base = T.alloc_var(T.int32, init=pid * token_group)
                if use_compact_groups:
                    while expert_id < num_experts and pid >= group_psum_ub[expert_id]:
                        expert_id += 1
                    expert_group_base = T.alloc_var(T.int32, init=0)
                    expert_token_base = T.alloc_var(T.int32, init=0)
                    if expert_id > 0:
                        expert_group_base = group_psum_ub[expert_id - 1]
                        expert_token_base = (psum_ub[expert_id - 1] + alignment - 1) // alignment * alignment
                    token_base = expert_token_base + (pid - expert_group_base) * token_group
                elif has_psum:
                    while expert_id < num_experts and token_base >= (psum_ub[expert_id] + alignment - 1) // alignment * alignment:
                        expert_id += 1
                if has_psum:
                    valid_group = token_base < num_expanded_tokens and expert_id < num_experts and token_base < psum_ub[expert_id]
                else:
                    valid_group = token_base < num_expanded_tokens

                if valid_group:
                    for token_offset in T.serial(token_group):
                        token_id = T.alloc_var(T.int32, init=token_base + token_offset)
                        valid = is_valid_token(token_id, expert_id if has_psum else 0, psum_ub if has_psum else None)
                        if valid:
                            weight = load_inputs(x, xl_ub, xr_ub, topk_weights, routed_scaling_factor, token_id)
                            with T.SimdVF():
                                vdiv_precision = 'ftz_true' if use_fast_certificate else 'exact'
                                cert_min, cert_max = compute_swiglu(
                                    xl_ub, xr_ub, values_ub, out_ub, weight, vdiv_precision, amax_ub, use_fast_certificate, pacc_ub, count_clamp
                                )
                                if use_fast_certificate:
                                    S.vsts(certificate_ub[token_offset, 0], S.vcmin(cert_min), dist='ONEPT_B32')
                                    S.vsts(certificate_ub[token_offset, 1], S.vcmax(cert_max), dist='ONEPT_B32')
                            finish_token(
                                values_ub, out_ub, amax_ub, inverse_ub, sf_dense_ub, out, out_sf, token_offset, token_id, count_ub, count_clamp
                            )

                    if use_fast_certificate:  # fall back to exact vdiv for unsafe tokens
                        for fallback_offset in T.serial(token_group):
                            token_id = T.alloc_var(T.int32, init=token_base + fallback_offset)
                            valid = is_valid_token(token_id, expert_id if has_psum else 0, psum_ub if has_psum else None)
                            unsafe = T.alloc_var(
                                T.bool,
                                init=(
                                    certificate_ub[fallback_offset, 0] <= certificate_error
                                    or certificate_ub[fallback_offset, 1] >= 0x80000 - certificate_error
                                ),
                            )
                            if valid and unsafe:
                                weight = load_inputs(x, xl_fb_ub, xr_fb_ub, topk_weights, routed_scaling_factor, token_id)
                                with T.SimdVF():
                                    # Clamp counts were already accumulated in the main path; fallback must set count_clamp to False
                                    _, _ = compute_swiglu(xl_fb_ub, xr_fb_ub, values_fb_ub, out_fb_ub, weight, 'exact', amax_ub, False, None, False)
                                finish_token(
                                    values_fb_ub, out_fb_ub, amax_ub, inverse_ub, sf_dense_ub, out, out_sf, fallback_offset, token_id, None, False
                                )

                if out_config.with_sf and out_config.use_tma_aligned_col_major_sf:
                    if valid_group:
                        transpose_packed_scales(sf_dense_ub, sf_ub, transpose_indices_ub)
                        output_sf = T.Tensor((sf_shape[0], sf_shape[1] // pack_factor), T.uint16, out_sf.data)
                        valid_token_end = psum_ub[expert_id] if has_psum else num_expanded_tokens
                        valid_tokens = T.alloc_var(T.int32, init=T.min(token_group, valid_token_end - token_base))
                        T.copy(sf_ub[:, 0:valid_tokens], output_sf[:, token_base : token_base + valid_tokens])

            if count_clamp:
                with T.SimtVF(threads=32):
                    lane = T.get_thread_binding(0)
                    for i in T.serial(3):
                        count = T.alloc_var(T.int64, init=T.int64(pacc_ub[i, lane]) + T.int64(pacc_ub[i, lane + 32]))
                        # Split at bit 16
                        count_low = T.warp_reduce_sum(T.int32(count & 0xFFFF))
                        count_high = T.warp_reduce_sum(T.int32(count >> 16))
                        if lane == 0:
                            T.atomic_add(clamped_count[i], (T.int64(count_high) << 16) + T.int64(count_low))
                    if lane == 0:
                        T.atomic_add(clamped_count[3], count_ub[0])

    return swiglu_forward_kernel

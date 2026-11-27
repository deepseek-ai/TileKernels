import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from tile_kernels.engram.utils import load_f32


@tilelang.jit(compile_flags=['--cce-disable-vf-stack-reserved-ubuf'])
def get_engram_gate_bwd_kernel_asc(
    hidden_size: int,
    scalar: float,
    clamp_value: float,
    hc_mult: int,
    has_image_token_mask: bool,
    has_value_norm: bool,
    x_dtype: T.dtype,
    wv_dtype: T.dtype,
    num_vec_cores: int,
):
    vec_size = 64
    num_accs = 2
    assert hc_mult == 4
    assert hidden_size % (vec_size * num_accs) == 0
    assert x_dtype in (T.bfloat16, T.float32)
    assert wv_dtype in (T.bfloat16, T.float32)
    num_slots = 2
    num_chunks = hidden_size // vec_size
    num_tokens = T.dynamic('num_tokens')

    def input_row(storage, slot=0):
        start = storage.shape[1] - hidden_size
        return storage[slot, start : start + hidden_size]

    def input_at(storage, slot, offset):
        start = storage.shape[1] - hidden_size
        return storage[slot, start + offset]

    @T.prim_func
    def engram_gate_bwd_kernel_asc(
        grad_out: T.Tensor[(num_tokens, hc_mult, hidden_size), x_dtype],
        hidden_states: T.Tensor[(num_tokens, hc_mult, hidden_size), x_dtype],
        kv: T.Tensor[(num_tokens, hc_mult + 1, hidden_size), T.bfloat16],
        weight_xk_fused: T.Tensor[(hc_mult, hidden_size), T.float32],
        weight_value: T.Tensor[(hidden_size,), wv_dtype],
        dot: T.Tensor[(num_tokens, hc_mult), T.float32],
        gate_score: T.Tensor[(num_tokens, hc_mult), T.float32],
        rstd_x: T.Tensor[(num_tokens, hc_mult), T.float32],
        rstd_kv: T.Tensor[(num_tokens, hc_mult + has_value_norm), T.float32],
        image_token_mask: T.Tensor[(num_tokens,), T.bool],
        grad_x: T.Tensor[(num_tokens, hc_mult, hidden_size), x_dtype],
        grad_kv: T.Tensor[(num_tokens, hc_mult + 1, hidden_size), T.bfloat16],
        grad_w_partial: T.Tensor[(num_vec_cores, hc_mult + has_value_norm, hidden_size), T.float32],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            weight_ub = T.alloc_shared((num_slots, hidden_size), T.float32)
            go_ub = T.alloc_shared((num_slots, hidden_size), x_dtype)
            k_ub = T.alloc_shared((num_slots, hidden_size), T.bfloat16)
            grad_w_ub = T.alloc_shared((num_slots, hidden_size), T.float32)
            grad_head_value_ub = T.alloc_shared((hidden_size,), T.float32)
            dot_ub = T.alloc_shared((8,), T.float32)
            gate_ub = T.alloc_shared((8,), T.float32)
            rstd_x_ub = T.alloc_shared((8,), T.float32)
            rstd_kv_ub = T.alloc_shared((8,), T.float32)

            grad_w_bf16_ub = T.view(grad_w_ub, (num_slots, hidden_size * 2), T.bfloat16)
            if x_dtype == T.float32:
                x_ub = grad_w_ub
            else:
                x_ub = grad_w_bf16_ub

            head_value_ub = T.alloc_shared((1, hidden_size), T.float32)
            if has_value_norm:
                proj_v_sum_ub = T.alloc_shared((8,), T.float32)
                v_ub = k_ub
                grad_wv_ub = weight_ub
                v_prepare_slot = 1
                if wv_dtype == T.float32:
                    wv_ub = grad_w_ub
                else:
                    wv_ub = grad_w_bf16_ub
            else:
                v_ub = T.view(head_value_ub, (1, hidden_size * 2), T.bfloat16)
                v_prepare_slot = 0

            buffer_versions = {
                weight_ub: 1,
                go_ub: 1,
                k_ub: 1,
                grad_w_ub: 1,
                grad_head_value_ub: 1,
                dot_ub: 1,
                gate_ub: 1,
                rstd_x_ub: 1,
                rstd_kv_ub: 1,
                head_value_ub: 1,
            }
            if has_value_norm:
                buffer_versions.update({proj_v_sum_ub: 1})
            T.annotate_buffer_versions(buffer_versions)

            with T.SimdVF():
                T.clear(grad_head_value_ub)
            zero_rows = T.StridedTensor((hc_mult + has_value_norm, hidden_size), (0, 1), T.float32, scope='shared', data=grad_head_value_ub.data)
            with T.Task():
                T.copy(zero_rows, grad_w_partial[core_id, :, :], l2_cache_ctrl='NORMAL_LV')
                T.ascend_pipe_barrier('PIPE_MTE3')

            for token_id in T.serial(core_id, num_tokens, num_vec_cores):
                # BF16 aliases in opposite physical slots do not overlap.
                if x_dtype == T.bfloat16:
                    T.assume_no_conflict(x_ub[0, hidden_size : 2 * hidden_size], grad_w_ub[1, :], level=0)
                    T.assume_no_conflict(x_ub[1, hidden_size : 2 * hidden_size], grad_w_ub[0, :], level=0)
                T.copy(grad_out[token_id, 0, :], go_ub[0, :], l2_cache_ctrl='NOTALLOC_KEEP')
                if has_image_token_mask and image_token_mask[token_id]:
                    with T.SimdVF():
                        zero = S.vdup(0.0, T.bfloat16)
                        for chunk_id in T.serial(hidden_size // 128):
                            S.vsts(v_ub[0, chunk_id * 128], zero, dist='NORM_B16')
                    for head_id in T.unroll(hc_mult, explicit=True):
                        if head_id != 0:
                            T.copy(grad_out[token_id, head_id, :], go_ub[0, :])
                        T.copy(go_ub[0, :], grad_x[token_id, head_id, :])
                    zero_kv_rows = T.StridedTensor((hc_mult + 1, hidden_size), (0, 1), T.bfloat16, scope='shared', data=v_ub.data)
                    T.copy(zero_kv_rows, grad_kv[token_id, :, :])
                else:
                    T.copy(kv[token_id, hc_mult, :], input_row(v_ub, v_prepare_slot))

                    with T.Task():
                        T.copy(dot[token_id, :], dot_ub[:hc_mult])
                        T.copy(gate_score[token_id, :], gate_ub[:hc_mult])
                        T.copy(rstd_x[token_id, :], rstd_x_ub[:hc_mult])
                        T.copy(rstd_kv[token_id, :], rstd_kv_ub[: hc_mult + has_value_norm])

                    if has_value_norm:
                        T.copy(weight_value, input_row(wv_ub, 1))
                        with T.SimdVF():
                            for chunk_id in T.serial(num_chunks):
                                offset = chunk_id * vec_size
                                v_f32 = load_f32(v_ub[v_prepare_slot, offset])
                                wv_f32 = load_f32(input_at(wv_ub, 1, offset))
                                weighted_v = S.vmul(v_f32, wv_f32)
                                S.vsts(head_value_ub[0, offset], weighted_v)

                    for head_id in T.unroll(hc_mult, explicit=True):
                        slot = head_id % num_slots
                        T.copy(weight_xk_fused[head_id, :], weight_ub[slot, :])
                        T.copy(hidden_states[token_id, head_id, :], input_row(x_ub, slot), l2_cache_ctrl='NOTALLOC_KEEP')
                        if head_id != 0:
                            T.copy(grad_out[token_id, head_id, :], go_ub[slot, :], l2_cache_ctrl='NOTALLOC_KEEP')
                        T.copy(kv[token_id, head_id, :], k_ub[slot, :], l2_cache_ctrl='NOTALLOC_KEEP')

                        with T.SimdVF():
                            grad_gate_acc = S.alloc_local((num_accs,), T.float32)
                            for acc_id in T.unroll(num_accs, explicit=True):
                                grad_gate_acc[acc_id] = S.vdup(0.0, T.float32)
                            gate = S.vld(gate_ub[head_id], dist='BRC_B32')
                            for chunk_group in T.serial(num_chunks // num_accs):
                                for acc_id in T.unroll(num_accs, explicit=True):
                                    offset = (chunk_group * num_accs + acc_id) * vec_size
                                    go_f32 = load_f32(go_ub[slot, offset])
                                    weighted_v = S.alloc_local((1,), T.float32)
                                    if not has_value_norm and head_id == 0:
                                        weighted_v[0] = load_f32(input_at(v_ub, 0, offset))
                                        S.vsts(head_value_ub[0, offset], weighted_v[0])
                                    else:
                                        weighted_v[0] = load_f32(head_value_ub[0, offset])
                                    S.vmula(grad_gate_acc[acc_id], go_f32, weighted_v[0])
                                    grad_head_value_acc = S.alloc_local((1,), T.float32)
                                    if head_id == 0:
                                        grad_head_value_acc[0] = S.vmul(gate, go_f32)
                                    else:
                                        grad_head_value_acc[0] = S.vld(grad_head_value_ub[offset])
                                        S.vmula(grad_head_value_acc[0], gate, go_f32)
                                    if not has_value_norm and head_id == hc_mult - 1:
                                        S.vsts(v_ub[0, offset], S.vcvt(grad_head_value_acc[0], T.bfloat16), dist='PK_B32')
                                    else:
                                        S.vsts(grad_head_value_ub[offset], grad_head_value_acc[0])

                            grad_gate = (
                                S.vmul(S.vdupv(S.vcadd(S.vadd(grad_gate_acc[0], grad_gate_acc[1]))), S.vld(rstd_kv_ub[hc_mult], dist='BRC_B32'))
                                if has_value_norm
                                else S.vdupv(S.vcadd(S.vadd(grad_gate_acc[0], grad_gate_acc[1])))
                            )
                            if has_value_norm:
                                proj_v_acc = S.alloc_local((1,), T.float32)
                                if head_id == 0:
                                    proj_v_acc[0] = S.vmul(gate, grad_gate)
                                else:
                                    proj_v_acc[0] = S.vld(proj_v_sum_ub[0], dist='BRC_B32')
                                    S.vmula(proj_v_acc[0], gate, grad_gate)
                                S.vsts(proj_v_sum_ub[0], proj_v_acc[0], dist='ONEPT_B32')

                            raw_dot = S.vld(dot_ub[head_id], dist='BRC_B32')
                            rstd_x_f32 = S.vld(rstd_x_ub[head_id], dist='BRC_B32')
                            rstd_k_f32 = S.vld(rstd_kv_ub[head_id], dist='BRC_B32')
                            dot_scale = S.vmuls(S.vmul(rstd_x_f32, rstd_k_f32), scalar)
                            abs_dot = S.vabs(raw_dot)
                            active_gate = S.vcmps(S.vmul(abs_dot, dot_scale), clamp_value, op='ge')
                            sqrt_grad = S.vmuls(S.vsqrt(S.vdiv(dot_scale, abs_dot, mask=active_gate)), 0.5)
                            grad_dot = S.vmul(S.vmul(grad_gate, S.vsub(gate, S.vmul(gate, gate))), sqrt_grad, mask=active_gate)
                            proj = S.vmuls(S.vmul(grad_dot, raw_dot), -1.0 / hidden_size)
                            proj_x = S.vmul(S.vmul(proj, rstd_x_f32), rstd_x_f32)
                            proj_k = S.vmul(S.vmul(proj, rstd_k_f32), rstd_k_f32)

                            for chunk_id in T.serial(num_chunks):
                                offset = chunk_id * vec_size
                                x_f32 = load_f32(input_at(x_ub, slot, offset))
                                k_f32 = load_f32(k_ub[slot, offset])
                                weight_f32 = S.vld(weight_ub[slot, offset])
                                grad_dot_x = S.vmul(grad_dot, x_f32)
                                grad_dot_k = S.vmul(grad_dot, k_f32)
                                grad_x_acc = S.alloc_local((1,), T.float32)
                                grad_k_acc = S.alloc_local((1,), T.float32)
                                grad_w_acc = S.alloc_local((1,), T.float32)
                                grad_x_acc[0] = load_f32(go_ub[slot, offset])
                                grad_k_acc[0] = S.vmul(weight_f32, grad_dot_x)
                                S.vmula(grad_x_acc[0], weight_f32, grad_dot_k)
                                S.vmula(grad_x_acc[0], proj_x, x_f32)
                                S.vmula(grad_k_acc[0], proj_k, k_f32)
                                grad_w_acc[0] = S.vmul(grad_dot_x, k_f32)
                                if x_dtype == T.bfloat16:
                                    S.vsts(go_ub[slot, offset], S.vcvt(grad_x_acc[0], T.bfloat16), dist='PK_B32')
                                else:
                                    S.vsts(go_ub[slot, offset], grad_x_acc[0])
                                S.vsts(k_ub[slot, offset], S.vcvt(grad_k_acc[0], T.bfloat16), dist='PK_B32')
                                S.vsts(grad_w_ub[slot, offset], grad_w_acc[0])
                        T.set_atomic('add', 'float32')
                        with T.Task():
                            T.copy(grad_w_ub[slot, :], grad_w_partial[core_id, head_id, :], l2_cache_ctrl='NORMAL_LV')
                            # ensure deterministic
                            if not has_value_norm and head_id == hc_mult - 1:
                                T.ascend_pipe_barrier('PIPE_MTE3')
                        T.set_atomic_none()
                        T.copy(go_ub[slot, :], grad_x[token_id, head_id, :])
                        T.copy(k_ub[slot, :], grad_kv[token_id, head_id, :])
                    if has_value_norm:
                        T.copy(weight_value, input_row(wv_ub, 0))
                        T.copy(kv[token_id, hc_mult, :], v_ub[0, :])
                        with T.SimdVF():
                            rstd_v = S.vld(rstd_kv_ub[hc_mult], dist='BRC_B32')
                            proj_v_sum = S.vld(proj_v_sum_ub[0], dist='BRC_B32')
                            neg_proj_v = S.vmuls(S.vmul(S.vmul(proj_v_sum, rstd_v), rstd_v), -1.0 / hidden_size)
                            for chunk_id in T.serial(num_chunks):
                                offset = chunk_id * vec_size
                                grad_head_value = S.vld(grad_head_value_ub[offset])
                                v_f32 = load_f32(v_ub[0, offset])
                                wv_f32 = load_f32(input_at(wv_ub, 0, offset))
                                scaled_grad_v = S.vmul(grad_head_value, rstd_v)
                                grad_wv = S.vmul(scaled_grad_v, v_f32)
                                S.vsts(grad_wv_ub[0, offset], grad_wv)
                                grad_v_acc = S.alloc_local((1,), T.float32)
                                grad_v_acc[0] = S.vmul(scaled_grad_v, wv_f32)
                                S.vmula(grad_v_acc[0], v_f32, neg_proj_v)
                                S.vsts(v_ub[0, offset], S.vcvt(grad_v_acc[0], T.bfloat16), dist='PK_B32')
                    T.copy(v_ub[0, :hidden_size], grad_kv[token_id, hc_mult, :])
                    if has_value_norm:
                        T.set_atomic('add', 'float32')
                        with T.Task():
                            T.copy(grad_wv_ub[0, :], grad_w_partial[core_id, hc_mult, :], l2_cache_ctrl='NORMAL_LV')
                            T.ascend_pipe_barrier('PIPE_MTE3')
                        T.set_atomic_none()

    return engram_gate_bwd_kernel_asc

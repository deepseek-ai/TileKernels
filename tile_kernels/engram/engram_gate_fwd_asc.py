import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from tile_kernels.config import get_max_ub_per_vector_core
from tile_kernels.engram.utils import load_f32


@tilelang.jit
def get_engram_gate_fwd_kernel_asc(
    hidden_size: int,
    eps: float,
    scalar: float,
    clamp_value: float,
    hc_mult: int,
    save_for_backward: bool,
    has_image_token_mask: bool,
    has_value_norm: bool,
    x_dtype: T.dtype,
    wv_dtype: T.dtype,
    inplace: bool,
    num_vec_cores: int,
):
    vec_size = 64
    num_accs = 2

    assert x_dtype in (T.bfloat16, T.float32)
    assert wv_dtype in (T.bfloat16, T.float32)
    assert hc_mult == 4
    assert hidden_size % (vec_size * num_accs) == 0
    assert not (inplace and save_for_backward), 'inplace destroys hidden_states, which backward reads'

    ub_bytes = get_max_ub_per_vector_core(use_simt=False)
    stats_bytes = 32
    if save_for_backward:
        stats_bytes *= 4 + has_value_norm
    wv_bytes = hidden_size * wv_dtype.bytes if has_value_norm else 0
    v_bytes = hidden_size * 2

    for heads_per_task in (2, 1):
        weight_bytes = heads_per_task * hidden_size * 4 + wv_bytes
        x_bytes = heads_per_task * hidden_size * x_dtype.bytes
        k_bytes = heads_per_task * hidden_size * 2
        stage_bytes = x_bytes + k_bytes + v_bytes + stats_bytes
        num_stages = min(3, (ub_bytes - weight_bytes) // stage_bytes)
        if num_stages >= 2:
            break

    assert num_stages > 0
    num_head_groups = hc_mult // heads_per_task
    assert num_vec_cores % num_head_groups == 0

    num_tokens = T.dynamic('num_tokens')
    num_chunks = hidden_size // vec_size

    @T.prim_func
    def engram_gate_fwd_kernel_asc(
        hidden_states: T.Tensor[(num_tokens, hc_mult, hidden_size), x_dtype],
        kv: T.Tensor[(num_tokens, hc_mult + 1, hidden_size), T.bfloat16],
        weight_xk_fused: T.Tensor[(hc_mult, hidden_size), T.float32],
        weight_value: T.Tensor[(hidden_size,), wv_dtype],
        image_token_mask: T.Tensor[(num_tokens,), T.bool],
        output: T.Tensor[(num_tokens, hc_mult, hidden_size), x_dtype],
        dot_out: T.Tensor[(num_tokens, hc_mult), T.float32],
        gate_score: T.Tensor[(num_tokens, hc_mult), T.float32],
        rstd_x: T.Tensor[(num_tokens, hc_mult), T.float32],
        rstd_kv: T.Tensor[(num_tokens, hc_mult + has_value_norm), T.float32],
    ):
        output_target = hidden_states if inplace else output
        with T.Kernel(num_vec_cores) as core_id:
            head_group = core_id % num_head_groups
            head_base = head_group * heads_per_task

            weight_ub = T.alloc_shared((heads_per_task, hidden_size), T.float32)
            if has_value_norm:
                wv_ub = T.alloc_shared((hidden_size,), wv_dtype)

            x_ub = T.alloc_shared((heads_per_task, hidden_size), x_dtype)
            k_ub = T.alloc_shared((heads_per_task, hidden_size), T.bfloat16)
            v_ub = T.alloc_shared((hidden_size,), T.bfloat16)

            gate_ub = T.alloc_shared((8,), T.float32)
            buffer_versions = {
                x_ub: num_stages,
                k_ub: num_stages,
                v_ub: num_stages,
                gate_ub: num_stages,
            }
            if save_for_backward:
                dot_ub = T.alloc_shared((8,), T.float32)
                rstd_x_ub = T.alloc_shared((8,), T.float32)
                rstd_k_ub = T.alloc_shared((8,), T.float32)
                buffer_versions.update({dot_ub: num_stages, rstd_x_ub: num_stages, rstd_k_ub: num_stages})
                if has_value_norm:
                    rstd_v_ub = T.alloc_shared((8,), T.float32)
                    buffer_versions.update({rstd_v_ub: num_stages})
            T.annotate_buffer_versions(buffer_versions)

            T.copy(weight_xk_fused[head_base : head_base + heads_per_task, :], weight_ub)
            if has_value_norm:
                T.copy(weight_value, wv_ub)

            for token_id, task_head_group in T.Persistent(
                [num_tokens, num_head_groups],
                num_vec_cores,
                core_id,
                group_size=num_head_groups,
                num_stages=num_stages,
            ):
                task_head_base = task_head_group * heads_per_task
                if has_image_token_mask and image_token_mask[token_id]:
                    if not inplace:
                        T.copy(
                            hidden_states[token_id, task_head_base : task_head_base + heads_per_task, :],
                            x_ub,
                            l2_cache_ctrl='NOTALLOC_KEEP',
                        )
                        T.copy(x_ub, output_target[token_id, task_head_base : task_head_base + heads_per_task, :])
                else:
                    T.copy(
                        hidden_states[token_id, task_head_base : task_head_base + heads_per_task, :],
                        x_ub,
                        l2_cache_ctrl='NOTALLOC_KEEP',
                    )
                    T.copy(
                        kv[token_id, task_head_base : task_head_base + heads_per_task, :],
                        k_ub,
                        l2_cache_ctrl='NOTALLOC_KEEP',
                    )
                    with T.SimdVF():
                        one = S.vdup(1.0, T.float32)
                        zero = S.vdup(0.0, T.float32)
                        sqrsum_x_acc = S.alloc_local((num_accs,), T.float32)
                        sqrsum_k_acc = S.alloc_local((num_accs,), T.float32)
                        dot_acc = S.alloc_local((num_accs,), T.float32)
                        mean_x = S.alloc_var(T.float32)
                        mean_k = S.alloc_var(T.float32)
                        eps_vec = S.vdup(eps, T.float32)
                        inv_hidden_vec = S.vdup(1.0 / hidden_size, T.float32)

                        for head_id in T.unroll(heads_per_task, explicit=True):
                            for acc_id in T.unroll(num_accs, explicit=True):
                                sqrsum_x_acc[acc_id] = zero
                                sqrsum_k_acc[acc_id] = zero
                                dot_acc[acc_id] = zero

                            for chunk_group in T.serial(num_chunks // num_accs):
                                for acc_id in T.unroll(num_accs, explicit=True):
                                    offset = (chunk_group * num_accs + acc_id) * vec_size
                                    x_f32 = load_f32(x_ub[head_id, offset])
                                    k_f32 = load_f32(k_ub[head_id, offset])
                                    weight_f32 = S.vld(weight_ub[head_id, offset])

                                    S.vmula(sqrsum_x_acc[acc_id], x_f32, x_f32)
                                    S.vmula(sqrsum_k_acc[acc_id], k_f32, k_f32)
                                    weighted_x = S.vmul(x_f32, weight_f32)
                                    S.vmula(dot_acc[acc_id], weighted_x, k_f32)

                            mean_x = S.vcadd(S.vadd(sqrsum_x_acc[0], sqrsum_x_acc[1]))
                            mean_k = S.vcadd(S.vadd(sqrsum_k_acc[0], sqrsum_k_acc[1]))
                            raw_dot = S.vcadd(S.vadd(dot_acc[0], dot_acc[1]))
                            S.vmadd(mean_x, inv_hidden_vec, eps_vec)
                            S.vmadd(mean_k, inv_hidden_vec, eps_vec)
                            rms_x = S.vsqrt(mean_x)
                            rms_k = S.vsqrt(mean_k)
                            rstd_x_f32 = S.vdiv(one, rms_x)
                            rstd_k_f32 = S.vdiv(one, rms_k)
                            normalized_dot = S.vmuls(S.vmul(S.vmul(raw_dot, rstd_x_f32), rstd_k_f32), scalar)
                            signed_root_abs = S.vsqrt(S.vmaxs(S.vabs(normalized_dot), clamp_value))
                            signed_root = S.vsel(
                                signed_root_abs,
                                S.vneg(signed_root_abs),
                                S.vcmps(normalized_dot, 0.0, op='ge'),
                            )
                            gate = S.vdiv(one, S.vadd(S.vexpdif(zero, signed_root), one))

                            S.vsts(gate_ub[head_id], gate, dist='ONEPT_B32')
                            if save_for_backward:
                                S.vsts(dot_ub[head_id], raw_dot, dist='ONEPT_B32')
                                S.vsts(rstd_x_ub[head_id], rstd_x_f32, dist='ONEPT_B32')
                                S.vsts(rstd_k_ub[head_id], rstd_k_f32, dist='ONEPT_B32')

                    T.copy(kv[token_id, hc_mult, :], v_ub)

                    with T.SimdVF():
                        if has_value_norm:
                            one = S.vdup(1.0, T.float32)
                            zero = S.vdup(0.0, T.float32)
                            sqrsum_v_acc = S.alloc_local((num_accs,), T.float32)
                            for acc_id in T.unroll(num_accs, explicit=True):
                                sqrsum_v_acc[acc_id] = zero
                            for chunk_group in T.serial(num_chunks // num_accs):
                                for acc_id in T.unroll(num_accs, explicit=True):
                                    offset = (chunk_group * num_accs + acc_id) * vec_size
                                    v_f32 = load_f32(v_ub[offset])
                                    S.vmula(sqrsum_v_acc[acc_id], v_f32, v_f32)
                            sqrsum_v = S.vdupv(S.vcadd(S.vadd(sqrsum_v_acc[0], sqrsum_v_acc[1])))
                            rms_v = S.vsqrt(S.vadds(S.vmuls(sqrsum_v, 1.0 / hidden_size), eps))
                            rstd_v_f32 = S.vdiv(one, rms_v)
                            if save_for_backward:
                                S.vsts(rstd_v_ub[0], rstd_v_f32, dist='ONEPT_B32')

                        gates_f32 = S.alloc_local((heads_per_task,), T.float32)
                        for head_id in T.unroll(heads_per_task, explicit=True):
                            gates_f32[head_id] = S.vld(gate_ub[head_id], dist='BRC_B32')
                            if has_value_norm:
                                gates_f32[head_id] = S.vmul(gates_f32[head_id], rstd_v_f32)

                        out_f32 = S.alloc_local((heads_per_task,), T.float32)
                        for chunk_id in T.serial(num_chunks):
                            offset = chunk_id * vec_size
                            v_f32 = load_f32(v_ub[offset])
                            if has_value_norm:
                                wv_f32 = load_f32(wv_ub[offset])
                            for head_id in T.unroll(heads_per_task, explicit=True):
                                out_f32[head_id] = load_f32(x_ub[head_id, offset])
                            for head_id in T.unroll(heads_per_task, explicit=True):
                                if has_value_norm:
                                    S.vmula(out_f32[head_id], S.vmul(gates_f32[head_id], wv_f32), v_f32)
                                else:
                                    S.vmula(out_f32[head_id], v_f32, gates_f32[head_id])
                            for head_id in T.unroll(heads_per_task, explicit=True):
                                if x_dtype == T.bfloat16:
                                    S.vsts(x_ub[head_id, offset], S.vcvt(out_f32[head_id], T.bfloat16), dist='PK_B32')
                                else:
                                    S.vsts(x_ub[head_id, offset], out_f32[head_id])

                    T.copy(x_ub, output_target[token_id, task_head_base : task_head_base + heads_per_task, :])
                    if save_for_backward:
                        T.copy(dot_ub[:heads_per_task], dot_out[token_id, task_head_base : task_head_base + heads_per_task])
                        T.copy(gate_ub[:heads_per_task], gate_score[token_id, task_head_base : task_head_base + heads_per_task])
                        T.copy(rstd_x_ub[:heads_per_task], rstd_x[token_id, task_head_base : task_head_base + heads_per_task])
                        T.copy(rstd_k_ub[:heads_per_task], rstd_kv[token_id, task_head_base : task_head_base + heads_per_task])
                        if has_value_norm and task_head_group == 0:
                            T.copy(rstd_v_ub[:1], rstd_kv[token_id, hc_mult : hc_mult + 1])

    return engram_gate_fwd_kernel_asc

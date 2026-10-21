import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S


@tilelang.jit
def get_engram_gate_bwd_kernel_asc(
    hidden_size: int,
    scalar: float,
    clamp_value: float,
    hc_mult: int,
    has_image_token_mask: bool,
    num_vec_cores: int,
):
    vec_size = 64
    bf16_vec_size = 128
    num_accs = 2
    num_partials = num_vec_cores // hc_mult
    v_slice_size = hidden_size // hc_mult

    assert hc_mult == 4
    assert num_vec_cores % hc_mult == 0
    assert hidden_size % (hc_mult * vec_size) == 0

    num_chunks = hidden_size // vec_size
    num_chunk_groups = num_chunks // num_accs
    v_slice_chunks = v_slice_size // vec_size
    num_bf16_chunks = hidden_size // bf16_vec_size
    v_slice_bf16_chunks = v_slice_size // bf16_vec_size
    num_tokens = T.dynamic('num_tokens')

    @T.prim_func
    def engram_gate_bwd_kernel_asc(
        grad_out: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        hidden_states: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        kv: T.Tensor[(num_tokens, hc_mult + 1, hidden_size), T.bfloat16],
        weight_fused: T.Tensor[(hc_mult, hidden_size), T.float32],
        dot: T.Tensor[(num_tokens, hc_mult), T.float32],
        gate: T.Tensor[(num_tokens, hc_mult), T.float32],
        rstd_x: T.Tensor[(num_tokens, hc_mult), T.float32],
        rstd_k: T.Tensor[(num_tokens, hc_mult), T.float32],
        image_token_mask: T.Tensor[(num_tokens,), T.bool],
        grad_x: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        grad_kv: T.Tensor[(num_tokens, hc_mult + 1, hidden_size), T.bfloat16],
        grad_w_partial: T.Tensor[(num_partials, hc_mult, hidden_size), T.float32],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            core_head = core_id % hc_mult
            go = T.alloc_shared((hidden_size,), T.bfloat16)
            other_go = T.alloc_shared((hc_mult - 1, v_slice_size), T.bfloat16)
            v = T.alloc_shared((hidden_size,), T.bfloat16)
            x = T.alloc_shared((hidden_size,), T.bfloat16)
            k = T.alloc_shared((hidden_size,), T.bfloat16)
            stats = T.alloc_shared((32,), T.float32)
            weight = T.alloc_shared((hidden_size,), T.float32)
            grad_w = T.alloc_shared((hidden_size,), T.float32)

            T.annotate_buffer_versions(
                {
                    go: 2,
                    other_go: 3,
                    v: 3,
                    x: 3,
                    k: 3,
                    stats: 3,
                }
            )
            T.copy(weight_fused[core_head, :], weight)
            with T.SimdVF():
                T.clear(grad_w)

            for token, task_head in T.Persistent(
                [num_tokens, hc_mult],
                num_vec_cores,
                core_id,
                group_size=hc_mult,
                num_stages=6,
            ):
                v_slice_begin = task_head * v_slice_size

                T.copy(grad_out[token, task_head, :], go)
                if has_image_token_mask and image_token_mask[token]:
                    with T.SimdVF():
                        zero_bf16 = S.vdup(0.0, T.bfloat16)
                        for chunk in range(num_bf16_chunks):
                            S.vsts(x[chunk * bf16_vec_size], zero_bf16, dist='NORM_B16')
                        for chunk in range(v_slice_bf16_chunks):
                            S.vsts(v[v_slice_begin + chunk * bf16_vec_size], zero_bf16, dist='NORM_B16')
                        if v_slice_size % bf16_vec_size != 0:
                            S.vsts(v[v_slice_begin + v_slice_bf16_chunks * bf16_vec_size], zero_bf16, dist='PK_B32')
                else:
                    with T.Task():
                        for head_delta in T.unroll(1, hc_mult, explicit=True):
                            other_head = (task_head + head_delta) % hc_mult
                            T.copy(
                                grad_out[
                                    token,
                                    other_head,
                                    v_slice_begin : v_slice_begin + v_slice_size,
                                ],
                                other_go[head_delta - 1, :],
                            )
                        T.copy(hidden_states[token, task_head, :], x, l2_cache_ctrl='NOTALLOC_KEEP')
                        T.copy(kv[token, task_head, :], k, l2_cache_ctrl='NOTALLOC_KEEP')
                        T.copy(kv[token, hc_mult, :], v)
                        T.copy(dot[token, :], stats[0:hc_mult])
                        T.copy(gate[token, :], stats[8 : 8 + hc_mult])
                        T.copy(rstd_x[token, :], stats[16 : 16 + hc_mult])
                        T.copy(rstd_k[token, :], stats[24 : 24 + hc_mult])

                    with T.SimdVF():
                        zero = S.vdup(0.0, T.float32)
                        dgate_acc = S.alloc_local((num_accs,), T.float32)
                        for acc_id in T.unroll(num_accs, explicit=True):
                            dgate_acc[acc_id] = zero

                        for chunk_group in range(num_chunk_groups):
                            for acc_id in T.unroll(num_accs, explicit=True):
                                offset = (chunk_group * num_accs + acc_id) * vec_size
                                go_f32 = S.vcvt(S.vld(go[offset], dist='UNPK_B16'), T.float32, part=0)
                                v_f32 = S.vcvt(S.vld(v[offset], dist='UNPK_B16'), T.float32, part=0)
                                S.vmula(dgate_acc[acc_id], go_f32, v_f32)
                        dgate = S.vdupv(S.vcadd(S.vadd(dgate_acc[0], dgate_acc[1])))

                        gate_value = S.vld(stats[8 + task_head], dist='BRC_B32')
                        raw_dot = S.vld(stats[task_head], dist='BRC_B32')
                        inv_rms_x = S.vld(stats[16 + task_head], dist='BRC_B32')
                        inv_rms_k = S.vld(stats[24 + task_head], dist='BRC_B32')
                        dot_scale = S.vmuls(S.vmul(inv_rms_x, inv_rms_k), scalar)
                        abs_raw_dot = S.vabs(raw_dot)
                        active_gate = S.vcmps(S.vmul(abs_raw_dot, dot_scale), clamp_value, op='ge')
                        sqrt_grad = S.vmuls(S.vsqrt(S.vdiv(dot_scale, abs_raw_dot, mask=active_gate)), 0.5)
                        gate_grad = S.vsub(gate_value, S.vmul(gate_value, gate_value))
                        beta = S.vmul(S.vmul(dgate, gate_grad), sqrt_grad, mask=active_gate)

                        neg_beta_raw_dot_over_h = S.vmuls(S.vmul(beta, raw_dot), -1.0 / hidden_size)
                        neg_dot_x = S.vmul(S.vmul(neg_beta_raw_dot_over_h, inv_rms_x), inv_rms_x)
                        neg_dot_k = S.vmul(S.vmul(neg_beta_raw_dot_over_h, inv_rms_k), inv_rms_k)

                        grad_v_acc = S.alloc_local((1,), T.float32)
                        other_gates = S.alloc_local((hc_mult - 1,), T.float32)
                        for head_delta in T.unroll(1, hc_mult, explicit=True):
                            other_head = (task_head + head_delta) % hc_mult
                            other_gates[head_delta - 1] = S.vld(stats[8 + other_head], dist='BRC_B32')

                        for chunk in range(v_slice_chunks):
                            local_offset = chunk * vec_size
                            v_offset = v_slice_begin + local_offset
                            own_go = S.vcvt(S.vld(go[v_offset], dist='UNPK_B16'), T.float32, part=0)
                            grad_v_acc[0] = S.vmul(own_go, gate_value)
                            for head_delta in T.unroll(1, hc_mult, explicit=True):
                                other_go_f32 = S.vcvt(
                                    S.vld(
                                        other_go[
                                            head_delta - 1,
                                            local_offset,
                                        ],
                                        dist='UNPK_B16',
                                    ),
                                    T.float32,
                                    part=0,
                                )
                                S.vmula(grad_v_acc[0], other_go_f32, other_gates[head_delta - 1])
                            S.vsts(v[v_offset], S.vcvt(grad_v_acc[0], T.bfloat16), dist='PK_B32')
                        accum = S.alloc_local((3,), T.float32)

                        for chunk in range(num_chunks):
                            offset = chunk * vec_size
                            go_f32 = S.vcvt(S.vld(go[offset], dist='UNPK_B16'), T.float32, part=0)
                            x_f32 = S.vcvt(S.vld(x[offset], dist='UNPK_B16'), T.float32, part=0)
                            k_f32 = S.vcvt(S.vld(k[offset], dist='UNPK_B16'), T.float32, part=0)
                            weight_f32 = S.vld(weight[offset])
                            beta_x = S.vmul(beta, x_f32)
                            beta_k = S.vmul(beta, k_f32)

                            accum[0] = go_f32
                            accum[1] = S.vmul(weight_f32, beta_x)
                            accum[2] = S.vld(grad_w[offset])
                            S.vmula(accum[0], weight_f32, beta_k)
                            S.vmula(accum[1], neg_dot_k, k_f32)
                            S.vmula(accum[2], beta_x, k_f32)
                            S.vmula(accum[0], neg_dot_x, x_f32)

                            S.vsts(go[offset], S.vcvt(accum[0], T.bfloat16), dist='PK_B32')
                            S.vsts(x[offset], S.vcvt(accum[1], T.bfloat16), dist='PK_B32')
                            S.vsts(grad_w[offset], accum[2])

                with T.Task():
                    T.copy(go, grad_x[token, task_head, :])
                    T.copy(x, grad_kv[token, task_head, :])
                    T.copy(
                        v[v_slice_begin : v_slice_begin + v_slice_size],
                        grad_kv[
                            token,
                            hc_mult,
                            v_slice_begin : v_slice_begin + v_slice_size,
                        ],
                    )

            T.ascend_pipe_barrier('PIPE_ALL')
            T.copy(grad_w, grad_w_partial[core_id // hc_mult, core_head, :])

    return engram_gate_bwd_kernel_asc

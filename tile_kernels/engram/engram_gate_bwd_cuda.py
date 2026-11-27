import math

import tilelang
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def get_engram_gate_bwd_kernel_cuda(
    hidden_size: int,
    scalar: float,
    num_persistent_blocks: int,
    clamp_value: float = 1e-6,
    hc_mult: int = 4,
    has_image_token_mask: bool = False,
    has_value_norm: bool = False,
    x_dtype: T.dtype = T.bfloat16,
    wv_dtype: T.dtype = T.bfloat16,
    use_pdl: bool = False,
):
    num_tokens = T.dynamic('num_tokens')
    assert x_dtype in (T.bfloat16, T.float32)
    assert wv_dtype in (T.bfloat16, T.float32)
    assert hc_mult == 4
    assert hidden_size % 128 == 0
    vec_size = math.gcd(16 // x_dtype.bytes, hidden_size // 32)
    for threads in range(512, 0, -32):
        if hidden_size % (threads * vec_size) == 0:
            break
    num_warps = threads // 32
    blk_d = threads * vec_size
    num_blk = hidden_size // blk_d
    row_layout = T.Fragment(
        (hidden_size,),
        forward_thread_fn=lambda i_d: i_d // vec_size % threads,
        forward_index_fn=lambda i_d: i_d // blk_d * vec_size + i_d % vec_size,
    )
    go_layout = T.Fragment(
        (hc_mult, hidden_size),
        forward_thread_fn=lambda i_h, i_d: i_d // vec_size % threads,
        forward_index_fn=lambda i_h, i_d: i_h * num_blk * vec_size + i_d // blk_d * vec_size + i_d % vec_size,
    )

    blk_layout = T.Fragment((blk_d,), forward_thread_fn=lambda i_d: i_d // vec_size)
    cp_async_annotations = {'no_implicit_async_commit_wait': True}

    @T.prim_func
    def engram_gate_bwd_kernel_cuda(
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
        grad_w_partial: T.Tensor[(num_persistent_blocks, hc_mult + has_value_norm, hidden_size), T.float32],
    ):
        with T.Kernel(num_persistent_blocks, threads=threads) as pid_p:
            if use_pdl:
                T.pdl_sync()
            thread_idx = T.get_thread_binding()
            warp_idx = thread_idx // 32
            lane_idx = thread_idx % 32

            go_smem = T.alloc_shared((hc_mult, hidden_size), x_dtype)
            x_smem = T.alloc_shared((2, hidden_size), x_dtype)
            kv_smem = T.alloc_shared((2, hidden_size), T.bfloat16)
            stats_smem = T.alloc_shared((2, num_warps, hc_mult), T.float32)
            stats_phase = T.alloc_var(T.int32, init=0)

            go_local = T.alloc_local((vec_size,), T.float32)
            v_local = T.alloc_local((vec_size,), T.float32)
            x_local = T.alloc_local((vec_size,), T.float32)
            k_local = T.alloc_local((vec_size,), T.float32)
            w_local = T.alloc_local((vec_size,), T.float32)
            grad_gate_acc_local = T.alloc_local((hc_mult, 2), T.float32)
            grad_gate_local = T.alloc_local((hc_mult,), T.float32)
            gate_local = T.alloc_local((hc_mult,), T.float32)
            grad_v_local = T.alloc_local((num_blk, vec_size), T.float32)
            grad_w_local = T.alloc_local((hc_mult, num_blk, vec_size), T.float32)

            if has_value_norm:
                v_cache_local = T.alloc_local((num_blk, vec_size), T.bfloat16)
                wv_smem = T.alloc_shared((hidden_size,), wv_dtype)
                wv_local = T.alloc_local((vec_size,), T.float32)
                rstd_v_local = T.alloc_local((1,), T.float32)
                proj_v_local = T.alloc_local((1,), T.float32)
                grad_wv_local = T.alloc_local((num_blk, vec_size), T.float32)
                T.clear(grad_wv_local)
                if pid_p < num_tokens:
                    T.async_copy(weight_value, wv_smem, loop_layout=row_layout)
            T.clear(grad_w_local)

            for i_s in T.serial(pid_p, num_tokens, num_persistent_blocks):
                if has_image_token_mask and image_token_mask[i_s]:
                    for i_h, i_d in T.Parallel(hc_mult, hidden_size):
                        grad_x[i_s, i_h, i_d] = grad_out[i_s, i_h, i_d]
                    for i_h, i_d in T.Parallel(hc_mult + 1, hidden_size):
                        grad_kv[i_s, i_h, i_d] = 0.0
                else:
                    T.async_copy(grad_out[i_s, :, :], go_smem, loop_layout=go_layout, annotations=cp_async_annotations)
                    T.async_copy(kv[i_s, hc_mult, :], kv_smem[1, :], loop_layout=row_layout, annotations=cp_async_annotations)
                    T.ptx_commit_group()

                    T.async_copy(hidden_states[i_s, 0, :], x_smem[0, :], loop_layout=row_layout, annotations=cp_async_annotations)
                    T.async_copy(kv[i_s, 0, :], kv_smem[0, :], loop_layout=row_layout, annotations=cp_async_annotations)
                    T.ptx_commit_group()

                    if has_value_norm:
                        rstd_v_local[0] = rstd_kv[i_s, hc_mult]
                        T.clear(proj_v_local)
                    T.clear(grad_gate_acc_local)
                    T.copy(gate_score[i_s, :], gate_local, disable_tma=True)
                    for i_b in T.unroll(num_blk):
                        for i_k in T.vectorized(vec_size):
                            grad_v_local[i_b, i_k] = 0.0

                    T.ptx_wait_group(1)
                    T.async_copy(hidden_states[i_s, 1, :], x_smem[1, :], loop_layout=row_layout, annotations=cp_async_annotations)
                    for i_b in T.unroll(num_blk):
                        sub_base = i_b * blk_d + thread_idx * vec_size
                        if has_value_norm:
                            for i_k in T.vectorized(vec_size):
                                v_cache_local[i_b, i_k] = kv_smem[1, sub_base + i_k]
                                wv_local[i_k] = wv_smem[sub_base + i_k]
                            for i_k in T.vectorized(vec_size):
                                v_local[i_k] = v_cache_local[i_b, i_k] * wv_local[i_k]
                        else:
                            for i_k in T.vectorized(vec_size):
                                v_local[i_k] = kv_smem[1, sub_base + i_k]
                        for i_h in T.unroll(hc_mult):
                            for i_k in T.vectorized(vec_size):
                                go_local[i_k] = go_smem[i_h, sub_base + i_k]
                            for i_k in T.vectorized(vec_size):
                                grad_gate_acc_local[i_h, i_k % 2] += go_local[i_k] * v_local[i_k]
                                grad_v_local[i_b, i_k] += gate_local[i_h] * go_local[i_k]
                        for i_k in T.vectorized(vec_size):
                            if has_value_norm:
                                grad_v_local[i_b, i_k] *= rstd_v_local[0]
                                grad_wv_local[i_b, i_k] += grad_v_local[i_b, i_k] * v_cache_local[i_b, i_k]
                                grad_v_local[i_b, i_k] *= wv_local[i_k]
                            else:
                                grad_kv[i_s, hc_mult, sub_base + i_k] = grad_v_local[i_b, i_k]
                        T.async_copy(
                            kv[i_s, 1, i_b * blk_d : (i_b + 1) * blk_d],
                            kv_smem[1, i_b * blk_d : (i_b + 1) * blk_d],
                            loop_layout=blk_layout,
                            annotations=cp_async_annotations,
                        )
                    T.ptx_commit_group()

                    for i_h in T.unroll(hc_mult):
                        grad_gate_local[i_h] = T.warp_reduce_sum(grad_gate_acc_local[i_h, 0] + grad_gate_acc_local[i_h, 1])
                    if lane_idx == 0:
                        for i_h in T.vectorized(hc_mult):
                            stats_smem[stats_phase, warp_idx, i_h] = grad_gate_local[i_h]
                    T.sync_threads()

                    for i_h in T.unroll(hc_mult):
                        grad_gate_local[i_h] = stats_smem[stats_phase, 0, i_h]
                        for i_warp in T.unroll(1, num_warps):
                            grad_gate_local[i_h] += stats_smem[stats_phase, i_warp, i_h]
                        if has_value_norm:
                            grad_gate_local[i_h] *= rstd_v_local[0]
                            proj_v_local[0] += gate_local[i_h] * grad_gate_local[i_h]
                    stats_phase ^= 1
                    if has_value_norm:
                        proj_v_local[0] *= rstd_v_local[0] * rstd_v_local[0] / hidden_size
                        for i_b in T.unroll(num_blk):
                            sub_base = i_b * blk_d + thread_idx * vec_size
                            for i_k in T.vectorized(vec_size):
                                grad_kv[i_s, hc_mult, sub_base + i_k] = grad_v_local[i_b, i_k] - v_cache_local[i_b, i_k] * proj_v_local[0]

                    for i_h in T.unroll(hc_mult):
                        i_stage = i_h % 2
                        T.ptx_wait_group(1)
                        dot_local = dot[i_s, i_h]
                        rstd_x_local = rstd_x[i_s, i_h]
                        rstd_k_local = rstd_kv[i_s, i_h]
                        scale_local = scalar * rstd_x_local * rstd_k_local
                        grad_dot_local = T.Select(
                            T.abs(dot_local) * scale_local >= clamp_value,
                            grad_gate_local[i_h] * gate_local[i_h] * (1.0 - gate_local[i_h]) * 0.5 * T.sqrt(scale_local / T.abs(dot_local)),
                            0.0,
                        )
                        proj_local = -grad_dot_local * dot_local / hidden_size
                        proj_x_local = proj_local * rstd_x_local * rstd_x_local
                        proj_k_local = proj_local * rstd_k_local * rstd_k_local
                        for i_b in T.unroll(num_blk):
                            sub_base = i_b * blk_d + thread_idx * vec_size
                            for i_k in T.vectorized(vec_size):
                                go_local[i_k] = go_smem[i_h, sub_base + i_k]
                                x_local[i_k] = x_smem[i_stage, sub_base + i_k]
                                k_local[i_k] = kv_smem[i_stage, sub_base + i_k]
                            if i_h + 2 < hc_mult:
                                T.async_copy(
                                    hidden_states[i_s, i_h + 2, i_b * blk_d : (i_b + 1) * blk_d],
                                    x_smem[i_stage, i_b * blk_d : (i_b + 1) * blk_d],
                                    loop_layout=blk_layout,
                                    annotations=cp_async_annotations,
                                )
                                T.async_copy(
                                    kv[i_s, i_h + 2, i_b * blk_d : (i_b + 1) * blk_d],
                                    kv_smem[i_stage, i_b * blk_d : (i_b + 1) * blk_d],
                                    loop_layout=blk_layout,
                                    annotations=cp_async_annotations,
                                )
                            for i_k in T.vectorized(vec_size):
                                w_local[i_k] = weight_xk_fused[i_h, sub_base + i_k]
                            for i_k in T.vectorized(vec_size):
                                grad_dot_w = grad_dot_local * w_local[i_k]
                                grad_x[i_s, i_h, sub_base + i_k] = go_local[i_k] + grad_dot_w * k_local[i_k] + proj_x_local * x_local[i_k]
                                grad_kv[i_s, i_h, sub_base + i_k] = grad_dot_w * x_local[i_k] + proj_k_local * k_local[i_k]
                                grad_w_local[i_h, i_b, i_k] += grad_dot_local * x_local[i_k] * k_local[i_k]
                        T.ptx_commit_group()

            for i_h in T.unroll(hc_mult):
                for i_b in T.unroll(num_blk):
                    sub_base = i_b * blk_d + thread_idx * vec_size
                    for i_k in T.vectorized(vec_size):
                        grad_w_partial[pid_p, i_h, sub_base + i_k] = grad_w_local[i_h, i_b, i_k]
            if has_value_norm:
                for i_b in T.unroll(num_blk):
                    sub_base = i_b * blk_d + thread_idx * vec_size
                    for i_k in T.vectorized(vec_size):
                        grad_w_partial[pid_p, hc_mult, sub_base + i_k] = grad_wv_local[i_b, i_k]
            T.ptx_wait_group(0)
            if use_pdl:
                T.pdl_trigger()

    return engram_gate_bwd_kernel_cuda

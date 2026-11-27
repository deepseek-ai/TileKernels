import math

import tilelang
from tilelang import language as T

from tile_kernels.utils import ceil_div, next_power_of_two


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def get_engram_gate_fwd_kernel_cuda(
    hidden_size: int,
    eps: float,
    scalar: float,
    clamp_value: float = 1e-6,
    hc_mult: int = 4,
    save_for_backward: bool = True,
    has_image_token_mask: bool = False,
    has_value_norm: bool = False,
    x_dtype: T.dtype = T.bfloat16,
    wv_dtype: T.dtype = T.bfloat16,
    inplace: bool = False,
    use_pdl: bool = False,
):
    """Forward kernel. When save_for_backward=True, saves dot/gate_score/rstd_x/rstd_kv for backward."""
    assert x_dtype in (T.bfloat16, T.float32)
    assert wv_dtype in (T.bfloat16, T.float32)
    assert not (inplace and save_for_backward), 'inplace destroys hidden_states, which backward reads'
    assert hidden_size % 128 == 0

    num_tokens = T.dynamic('num_tokens')
    # Value normalization keeps V live across both passes. Favor wide vectors
    # while bounding per-thread live inputs and providing enough issuing warps.
    row_threads = next_power_of_two(ceil_div(hidden_size, 48 if has_value_norm else 80))
    issue_threads = min(128, next_power_of_two(ceil_div(hidden_size, 32)))
    threads = max(64, math.gcd(min(1024, max(row_threads, issue_threads)), hidden_size // 8))
    vec_size = math.gcd(8, hidden_size // threads)
    # A 32-byte stride produces conflicts between 16-byte shared loads.
    swizzle = vec_size * wv_dtype.bytes > 16
    # Occupancy hint: round the per-thread tile up and account for the extra
    # live data from value normalization and FP32 inputs.
    # An overly tight hint spills longer rows; unrestricted short rows can lose a CTA.
    tile_registers = next_power_of_two(hidden_size // threads)
    register_budget = max(72, (2 + has_value_norm + (x_dtype == T.float32)) * tile_registers + 8)
    resident_ctas = max(1, 512 // threads, min(16, 65536 // (threads * register_budget)))
    num_warps = threads // 32
    blk_d = threads * vec_size
    num_blk = hidden_size // blk_d
    num_stats = 3 + has_value_norm

    copy_vec = min(vec_size, 16 // wv_dtype.bytes)
    elem_log2 = int(math.log2(wv_dtype.bytes))

    @T.macro
    def load_fp32(src, offset, dst):
        if src.dtype == T.bfloat16:
            # Expand packed pairs exactly, preserving the compact input storage.
            words = T.view(src, (math.prod(src.shape) // 2,), T.uint32)
            for j in T.unroll(vec_size // 2):
                packed = words[offset // 2 + j]
                dst[2 * j] = T.reinterpret(packed << 16, T.float32)
                dst[2 * j + 1] = T.reinterpret(packed & T.uint32(0xFFFF0000), T.float32)
        else:
            values = T.reshape(src, (math.prod(src.shape),))
            for j in T.vectorized(vec_size):
                dst[j] = values[offset + j]

    @T.macro
    def swizzle_weight_index(index):
        return index ^ ((((index >> (7 - elem_log2)) & 1) << (4 - elem_log2)) if swizzle else 0)

    @T.prim_func
    def engram_gate_fwd_kernel_cuda(
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
        with T.Kernel(num_tokens * hc_mult, threads=threads) as pid:
            T.annotate_min_blocks_per_sm(resident_ctas)
            pid_h = pid % hc_mult
            pid_s = pid // hc_mult
            if use_pdl:
                T.pdl_sync()

            thread_idx = T.get_thread_binding()
            warp_idx = thread_idx // 32
            lane_idx = thread_idx % 32

            # Preserve input storage precision; all reductions and output arithmetic are FP32.
            x_local = T.alloc_local((num_blk, vec_size), x_dtype)
            v_local = T.alloc_local((num_blk, vec_size), T.bfloat16)
            k_local = T.alloc_local((vec_size,), T.float32)
            w_local = T.alloc_local((vec_size,), T.float32)
            k_bits = T.alloc_local((vec_size,), T.bfloat16)
            xf = T.alloc_local((vec_size,), T.float32)
            vf = T.alloc_local((vec_size,), T.float32)

            # Statistics: sum(x^2), sum(k^2), weighted dot, optional sum(v^2).
            accum = T.alloc_local((num_stats, 2), T.float32)
            stats = T.alloc_local((num_stats,), T.float32)
            stats_smem = T.alloc_shared((num_warps, num_stats), T.float32)

            rstd_x_local = T.alloc_var(T.float32)
            rstd_k_local = T.alloc_var(T.float32)
            gate_score_local = T.alloc_var(T.float32)
            if has_value_norm:
                rstd_v_local = T.alloc_var(T.float32)
                # A contiguous inner dimension lets TileLang vectorize shared loads.
                wv_smem = T.alloc_shared((hidden_size // copy_vec, copy_vec), wv_dtype)
                wv_local = T.alloc_local((vec_size,), wv_dtype)

            if has_image_token_mask and image_token_mask[pid_s]:
                if not inplace:
                    for i_d in T.Parallel(hidden_size):
                        output_target[pid_s, pid_h, i_d] = hidden_states[pid_s, pid_h, i_d]
            else:
                if has_value_norm:
                    # Adjacent lanes copy adjacent chunks; ownership stays within each warp.
                    # Keep the explicit XOR address: automatic layouts alter the copy schedule.
                    for cb in T.unroll(num_blk):
                        for part in T.unroll(vec_size // copy_vec):
                            cd = cb * blk_d + warp_idx * (32 * vec_size) + part * 32 * copy_vec + lane_idx * copy_vec
                            sd = swizzle_weight_index(cd)
                            T.ptx_cp_async(
                                wv_smem.access_ptr('w', offset=sd, extent=copy_vec),
                                weight_value.access_ptr('r', offset=cd, extent=copy_vec),
                                copy_vec,
                            )
                    T.ptx_commit_group()
                for i in T.unroll(num_stats, explicit=True):
                    for j in T.vectorized(2):
                        accum[i, j] = 0.0

                for i_b in T.unroll(num_blk):
                    sub_base = i_b * blk_d + thread_idx * vec_size
                    for i_k in T.vectorized(vec_size):
                        x_local[i_b, i_k] = hidden_states[pid_s, pid_h, sub_base + i_k]
                        v_local[i_b, i_k] = kv[pid_s, hc_mult, sub_base + i_k]
                        k_bits[i_k] = kv[pid_s, pid_h, sub_base + i_k]
                        w_local[i_k] = weight_xk_fused[pid_h, sub_base + i_k]
                    load_fp32(x_local, i_b * vec_size, xf)
                    load_fp32(k_bits, 0, k_local)
                    if has_value_norm:
                        load_fp32(v_local, i_b * vec_size, vf)
                    for i_k in T.vectorized(vec_size):
                        accum[0, i_k % 2] += xf[i_k] * xf[i_k]
                        accum[1, i_k % 2] += k_local[i_k] * k_local[i_k]
                        if has_value_norm:
                            accum[3, i_k % 2] += vf[i_k] * vf[i_k]
                        accum[2, i_k % 2] += xf[i_k] * w_local[i_k] * k_local[i_k]
                # Explicitly unroll the statistics axis to keep long rows fast.
                for i in T.unroll(num_stats, explicit=True):
                    stats[i] = T.warp_reduce_sum(accum[i, 0] + accum[i, 1])
                if lane_idx == 0:
                    for i in T.unroll(num_stats, explicit=True):
                        stats_smem[warp_idx, i] = stats[i]
                T.sync_threads()
                for i in T.unroll(num_stats, explicit=True):
                    stats[i] = stats_smem[0, i]
                for i_warp in T.unroll(1, num_warps):
                    for i in T.unroll(num_stats, explicit=True):
                        stats[i] += stats_smem[i_warp, i]
                rstd_x_local = T.rsqrt(stats[0] / hidden_size + eps)
                rstd_k_local = T.rsqrt(stats[1] / hidden_size + eps)
                if has_value_norm:
                    rstd_v_local = T.rsqrt(stats[3] / hidden_size + eps)
                gate_score_local = stats[2] * rstd_x_local * rstd_k_local * scalar
                gate_score_local = T.sigmoid(T.copysign(T.sqrt(T.max(T.abs(gate_score_local), clamp_value)), gate_score_local))
                output_scale = gate_score_local * rstd_v_local if has_value_norm else gate_score_local
                if has_value_norm:
                    T.ptx_wait_group(0)
                    if copy_vec != vec_size:
                        # The coalesced copy may assign a value to a different lane.
                        T.sync_warp()

                for i_b in T.unroll(num_blk):
                    sub_base = i_b * blk_d + thread_idx * vec_size
                    if has_value_norm:
                        for part in T.unroll(vec_size // copy_vec):
                            cd = sub_base + part * copy_vec
                            sd = swizzle_weight_index(cd)
                            for j in T.vectorized(copy_vec):
                                wv_local[part * copy_vec + j] = wv_smem[sd // copy_vec, j]
                    load_fp32(x_local, i_b * vec_size, xf)
                    load_fp32(v_local, i_b * vec_size, vf)
                    for i_k in T.vectorized(vec_size):
                        if has_value_norm:
                            output_target[pid_s, pid_h, sub_base + i_k] = xf[i_k] + output_scale * wv_local[i_k] * vf[i_k]
                        else:
                            output_target[pid_s, pid_h, sub_base + i_k] = xf[i_k] + output_scale * vf[i_k]
                # Keep the vector output loop ahead of scalar statistics stores.
                if save_for_backward and thread_idx == 0:
                    dot_out[pid_s, pid_h] = stats[2]
                    gate_score[pid_s, pid_h] = gate_score_local
                    rstd_x[pid_s, pid_h] = rstd_x_local
                    rstd_kv[pid_s, pid_h] = rstd_k_local
                    if has_value_norm and pid_h == 0:
                        rstd_kv[pid_s, hc_mult] = rstd_v_local
            if use_pdl:
                T.pdl_trigger()

    return engram_gate_fwd_kernel_cuda

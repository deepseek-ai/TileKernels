import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from tile_kernels.moe.sqrt_softplus import sqrt_softplus_backward_asc
from tile_kernels.utils import align


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_OUT_OF_BOUND_WARNING: True})
def get_moe_topk_gate_backward_kernel_asc(num_topk: int, num_routed_experts: int, aux_exists: bool, num_vec_cores: int):
    num_stages = 2
    vector_length = 64  # SIMD vector length (float32 lanes per vreg)
    max_num_topk = 32  # Keep the same top-k limit as forward.

    # Give each token a power-of-two lane group so XOR reductions stay within that token.
    reduction_steps = (num_topk - 1).bit_length()
    topk_width = 1 << reduction_steps
    rows_per_vec = vector_length // topk_width
    idx_stride = align(num_topk * 2, 8)  # int64 indices viewed as pairs of int32.
    weight_stride = align(num_topk, 8)
    assert num_topk <= max_num_topk
    num_aligned_experts = align(num_routed_experts, vector_length)
    num_vregs = num_aligned_experts // vector_length
    experts_aligned = num_routed_experts == num_aligned_experts
    if num_aligned_experts <= 256 or (not aux_exists and num_aligned_experts <= 384):
        block_m = 32
    else:
        block_m = 16

    num_tokens = T.dynamic('num_tokens')
    num_seqs = T.dynamic('num_seqs')

    @T.prim_func
    def moe_topk_gate_backward_kernel_asc(
        scores: T.Tensor[(num_tokens, num_routed_experts), T.float32],
        topk_idx: T.Tensor[(num_tokens, num_topk), T.int64],
        topk_weights: T.Tensor[(num_tokens, num_topk), T.float32],
        grad_topk_weights: T.Tensor[(num_tokens, num_topk), T.float32],
        mask: T.Tensor[(num_tokens,), T.bool],
        grad_scores_sum: T.Tensor[(num_seqs, num_routed_experts), T.float32],
        grad_logits: T.Tensor[(num_tokens, num_routed_experts), T.float32],
        routed_scaling_factor: T.float32,
    ):
        with T.Kernel(num_vec_cores) as core_id:
            scores_ub = T.alloc_shared((block_m, num_aligned_experts), T.float32)
            grad_ub = T.alloc_shared((block_m, num_aligned_experts), T.float32)
            idx_ub = T.alloc_shared((block_m, idx_stride), T.int32)
            weight_ub = T.alloc_shared((block_m, weight_stride), T.float32)
            grad_weight_ub = T.alloc_shared((block_m, weight_stride), T.float32)

            topk_idx_view = T.view(topk_idx, (num_tokens, num_topk * 2), T.int32)
            versions = {
                scores_ub: num_stages,
                grad_ub: num_stages,
                idx_ub: num_stages,
                weight_ub: num_stages,
                grad_weight_ub: num_stages,
            }
            if aux_exists:
                aux_grad_ub = T.alloc_shared((block_m, num_aligned_experts), T.float32)
                mask_ub = T.alloc_shared((block_m,), T.bool)
                versions.update({aux_grad_ub: num_stages, mask_ub: num_stages})
            T.annotate_buffer_versions(versions)

            num_rows = T.min(block_m, T.max(1, T.ceildiv(num_tokens, num_vec_cores)))
            for tile_id in T.Persistent(
                [T.ceildiv(num_tokens, num_rows)],
                num_vec_cores,
                core_id,
                num_stages=num_stages,
                annotations={'multi_buffer_eligible': [grad_ub]},
            ):
                row_start = tile_id * num_rows
                tile_rows = T.min(num_rows, num_tokens - row_start)
                T.copy(scores[row_start : row_start + num_rows, :], scores_ub[:num_rows, :num_routed_experts])
                T.copy(topk_idx_view[row_start : row_start + num_rows, :], idx_ub[:num_rows, : num_topk * 2])
                T.copy(topk_weights[row_start : row_start + num_rows, :], weight_ub[:num_rows, :num_topk])
                T.copy(grad_topk_weights[row_start : row_start + num_rows, :], grad_weight_ub[:num_rows, :num_topk])
                if aux_exists:
                    tokens_per_seq = num_tokens // num_seqs
                    seq_first = row_start // tokens_per_seq
                    seq_span = (row_start + tile_rows - 1) // tokens_per_seq - seq_first + 1
                    T.copy(mask[row_start : row_start + num_rows], mask_ub[:num_rows])
                    T.copy(grad_scores_sum[seq_first : seq_first + seq_span, :], aux_grad_ub[:seq_span, :num_routed_experts])

                with T.SimdVF():
                    lanes = S.vci(0, T.int32)
                    zero = S.vdup(0.0, T.float32)
                    local_row = S.vshrs(lanes, reduction_steps)
                    local_k = S.vand(lanes, S.vdup(topk_width - 1, T.int32))
                    idx_offsets = T.reinterpret(S.vadd(S.vmuls(local_row, idx_stride), S.vmuls(local_k, 2)), 'uint32x64')
                    weight_offsets = T.reinterpret(S.vadd(S.vmuls(local_row, weight_stride), local_k), 'uint32x64')
                    score_row_offsets = S.vmuls(local_row, num_aligned_experts)
                    for j in T.serial(tile_rows * num_vregs):
                        S.vsts(grad_ub[0, j * vector_length], zero)
                    S.mem_bar('VST_VST')

                    for row_base in T.serial(0, tile_rows, rows_per_vec):
                        valid_rows = S.vcmps(S.vadds(local_row, row_base), tile_rows, op='lt')
                        topk_lanes = S.vcmps(local_k, num_topk, valid_rows, op='lt')
                        idx = S.vgather2(idx_ub[row_base, 0], idx_offsets, topk_lanes)
                        valid_topk = S.vcmps(idx, 0, topk_lanes, op='ge')
                        score_offsets = T.reinterpret(S.vadd(score_row_offsets, idx), 'uint32x64')
                        chosen_scores = S.vgather2(scores_ub[row_base, 0], score_offsets, valid_topk)
                        weights = S.vgather2(weight_ub[row_base, 0], weight_offsets, valid_topk)
                        grad_weights = S.vgather2(grad_weight_ub[row_base, 0], weight_offsets, valid_topk)
                        denom = T.alloc_var(chosen_scores.dtype, init=chosen_scores)
                        dot = T.alloc_var(weights.dtype, init=S.vmul(weights, grad_weights))
                        for step in T.unroll(reduction_steps, explicit=True):
                            partner = S.vxor(lanes, S.vdup(1 << (reduction_steps - 1 - step), T.int32))
                            denom = S.vadd(denom, S.vselr(denom, partner))
                            dot = S.vadd(dot, S.vselr(dot, partner))
                        # For w_i = scale*s_i/sum(s), dL/ds_i = (scale*g_i - sum(w*g))/sum(s).
                        numerator = S.vsub(S.vmuls(grad_weights, routed_scaling_factor), dot)
                        delta = S.vdiv(numerator, S.vadds(denom, 1e-20), precision='ftz_true')

                        # Fixed routing may repeat experts; sum their gradients before a single scatter write.
                        combined = T.alloc_var(zero.dtype, init=zero)
                        last_lane = T.alloc_var(lanes.dtype, init=S.vdup(-1, T.int32))
                        for k in T.serial(num_topk):
                            selector = S.vadds(S.vmuls(local_row, topk_width), k)
                            same = S.vcmp(idx, S.vselr(idx, selector), op='eq')
                            combined = S.vadd(combined, S.vsel(S.vselr(delta, selector), zero, same))
                            last_lane = S.vsel(S.vdup(k, T.int32), last_lane, same)
                        unique = S.vcmp(local_k, last_lane, valid_topk, op='eq')
                        if not aux_exists:
                            combined = S.vmul(combined, sqrt_softplus_backward_asc(chosen_scores))
                        S.vscatter(combined, grad_ub[row_base, 0], score_offsets, unique)
                    S.mem_bar('VST_VLD')

                if aux_exists:
                    with T.SimdVF():
                        lanes = S.vci(0, T.int32)
                        zero = S.vdup(0.0, T.float32)
                        one = S.vdup(1.0, T.float32)
                        hrow = T.alloc_var(T.int32, init=0)
                        boundary = T.alloc_var(T.int32, init=(seq_first + 1) * tokens_per_seq - row_start)
                        for row in T.serial(tile_rows):
                            if row >= boundary:
                                hrow += 1
                                boundary += tokens_per_seq
                            score_sum = T.alloc_var(zero.dtype, init=zero)
                            aux_dot = T.alloc_var(zero.dtype, init=zero)
                            for j in T.serial(num_vregs):
                                score = S.vld(scores_ub[row, j * vector_length])
                                aux_grad = S.vld(aux_grad_ub[hrow, j * vector_length])
                                if not experts_aligned:
                                    valid = S.vcmps(S.vadds(lanes, j * vector_length), num_routed_experts, op='lt')
                                    score = S.vsel(score, zero, valid)
                                    aux_grad = S.vsel(aux_grad, zero, valid)
                                score_sum = S.vadd(score_sum, score)
                                aux_dot = S.vadd(aux_dot, S.vmul(aux_grad, score))
                            inv_score_sum = T.alloc_var(one.dtype, init=S.vdiv(one, S.vdupv(S.vcadd(score_sum)), precision='ftz_true'))
                            aux_active = S.vcmps(S.vdup(T.cast(mask_ub[row], T.int32), T.int32), 0, op='ne')
                            inv_score_sum = S.vsel(inv_score_sum, zero, aux_active)
                            aux_bias = S.vneg(S.vmul(S.vmul(S.vdupv(S.vcadd(aux_dot)), inv_score_sum), inv_score_sum))
                            for j in T.serial(num_vregs):
                                score = S.vld(scores_ub[row, j * vector_length])
                                aux_grad = S.vld(aux_grad_ub[hrow, j * vector_length])
                                grad_score = S.vadd(S.vld(grad_ub[row, j * vector_length]), S.vadd(S.vmul(aux_grad, inv_score_sum), aux_bias))
                                grad_logit = S.vmul(grad_score, sqrt_softplus_backward_asc(score))
                                S.vsts(grad_ub[row, j * vector_length], grad_logit)

                T.copy(grad_ub[:num_rows, :num_routed_experts], grad_logits[row_start : row_start + num_rows, :])

    return moe_topk_gate_backward_kernel_asc

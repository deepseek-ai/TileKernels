import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from tile_kernels.utils import align, ceil_div


@tilelang.jit
def get_topk_gate_kernel_asc(num_experts: int, num_topk: int, num_vec_cores: int):
    num_stages = 6
    vector_length = 64  # SIMD vector length (float32 lanes per vreg)

    # Keep up to six score vectors in registers; larger rows use the UB path.
    max_vregs_per_group = 6
    num_vregs_per_group = ceil_div(min(num_experts, max_vregs_per_group * vector_length), vector_length)
    experts_per_group = num_vregs_per_group * vector_length
    num_groups = ceil_div(num_experts, experts_per_group)
    num_aligned_experts = num_groups * experts_per_group

    # Round up to a multiple of vector_length for UB alignment of the index output.
    num_aligned_topk = align(num_topk, vector_length)

    num_tokens = T.dynamic('num_tokens')

    @T.prim_func
    def topk_gate_kernel_asc(
        scores: T.Tensor[(num_tokens, num_experts), T.float32],
        topk_idx: T.Tensor[(num_tokens, num_topk), T.int64],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            scores_ub = T.alloc_shared((num_aligned_experts,), T.float32)
            out_ub = T.alloc_shared((num_aligned_topk,), T.int32)
            invalid_logits_ub = T.alloc_shared((vector_length,), T.int32)
            topk_idx_view = T.view(topk_idx, (num_tokens, num_topk * 2), T.int32)
            T.annotate_buffer_versions({scores_ub: num_stages, out_ub: num_stages, invalid_logits_ub: 1})

            with T.SimdVF():
                T.clear(out_ub)
                T.clear(invalid_logits_ub)

            for w in T.Pipelined(
                T.ceildiv(num_tokens, num_vec_cores),
                num_stages=num_stages,
                annotations={'multi_buffer_eligible': [scores_ub, out_ub]},
            ):
                row = w * num_vec_cores + core_id
                if row < num_tokens:
                    T.copy(scores[row, :], scores_ub[:num_experts])

                    with T.SimdVF():
                        mask_32b = S.pset(32, 'PAT_ALL')
                        one = S.pset(32, 'PAT_VL1')

                        neg_inf = S.vdup(-T.infinity(T.float32), T.float32, mask_32b)
                        int_max = S.vdup(T.max_value(T.int32), T.int32, mask_32b)

                        # Allocate score and index vectors for each register in the group.
                        scores_vec = S.alloc_local((num_vregs_per_group,), T.float32)
                        index_vec = S.alloc_local((num_vregs_per_group,), T.int32)

                        tree_score = S.alloc_local((num_vregs_per_group,), T.float32)
                        tree_index = S.alloc_local((num_vregs_per_group,), T.int32)
                        lanes = S.vci(0, T.int32)
                        zero = S.vdup(0.0, T.float32)
                        zeros = S.vdup(0, T.int32)
                        invalid = T.alloc_var(T.int32x64, init=S.vld(invalid_logits_ub[0]))
                        logit = S.alloc_var(T.float32)
                        for j in T.unroll(ceil_div(num_experts, vector_length), explicit=True):
                            logit = S.vld(scores_ub[j * vector_length])
                            if (j + 1) * vector_length > num_experts:
                                valid = S.vcmps(S.vadds(lanes, j * vector_length), num_experts, op='lt')
                                logit = S.vsel(logit, zero, valid)
                            nonfinite = S.vcmps(S.vsub(logit, logit), 0.0, op='ne')
                            invalid = S.vmax(invalid, S.vsel(S.vdup(1, T.int32), zeros, nonfinite))
                        S.vsts(invalid_logits_ub[0], S.vcmax(invalid), one)

                        if num_groups == 1:
                            # Fast path: single group.
                            for i in T.unroll(num_vregs_per_group, explicit=True):
                                scores_vec[i] = S.vld(scores_ub[i * vector_length])
                                index_vec[i] = S.vci(T.int32(i * vector_length), T.int32)

                            num_exp = S.vdup(T.int32(num_experts), T.int32, mask_32b)
                            for i in T.unroll(num_vregs_per_group, explicit=True):
                                if (i + 1) * vector_length > num_experts:
                                    scores_vec[i] = S.vsel(scores_vec[i], neg_inf, S.vcmp(index_vec[i], num_exp, mask_32b, 'lt'))

                            acc_max = S.alloc_var(T.float32)
                            acc_min_idx = S.alloc_var(T.int32)
                            for k in T.serial(num_topk):
                                # Merge lane-local maxima with their smallest indices.
                                for i in T.unroll(num_vregs_per_group, explicit=True):
                                    tree_score[i] = scores_vec[i]
                                    tree_index[i] = index_vec[i]
                                # At most six registers need three pairwise merge levels.
                                for level in T.unroll(3, explicit=True):
                                    for i in T.unroll(num_vregs_per_group, explicit=True):
                                        if i % (2 << level) == 0 and i + (1 << level) < num_vregs_per_group:
                                            # Left subtree contains smaller indices; strict > preserves ties.
                                            greater = S.vcmp(tree_score[i + (1 << level)], tree_score[i], mask_32b, 'gt')
                                            tree_index[i] = S.vsel(tree_index[i + (1 << level)], tree_index[i], greater)
                                            tree_score[i] = S.vmax(tree_score[i], tree_score[i + (1 << level)], mask_32b)
                                acc_max = tree_score[0]
                                acc_min_idx = tree_index[0]
                                mx = S.vdupv(S.vcmax(acc_max, mask_32b), mask_32b)
                                acc_min_idx = S.vsel(acc_min_idx, int_max, S.vcmp(acc_max, mx, mask_32b, 'eq'))
                                reduced_idx = S.vcmin(acc_min_idx, mask_32b)
                                S.vsts(out_ub[2 * k], reduced_idx, one, 'ONEPT_B32')
                                idx = S.vdupv(reduced_idx, mask_32b)

                                # Mask out the winner lane in each vreg so the next k picks the next best.
                                for i in T.unroll(num_vregs_per_group, explicit=True):
                                    scores_vec[i] = S.vsel(neg_inf, scores_vec[i], S.vcmp(index_vec[i], idx, mask_32b, 'eq'))
                        else:
                            # Slow path: multi-group.
                            num_exp = S.vdup(T.int32(num_experts), T.int32, mask_32b)
                            acc = S.alloc_var(T.float32)
                            acc_idx = S.alloc_var(T.int32)

                            for k in T.serial(num_topk):
                                # Track each lane's best score and smallest index in one scan.
                                acc = neg_inf
                                acc_idx = int_max
                                for group_id in T.serial(num_groups):
                                    base = group_id * experts_per_group
                                    for i in T.unroll(num_vregs_per_group, explicit=True):
                                        offset = base + i * vector_length
                                        index_vec[i] = S.vci(T.int32(offset), T.int32)
                                        scores_vec[i] = S.vsel(S.vld(scores_ub[offset]), neg_inf, S.vcmp(index_vec[i], num_exp, mask_32b, 'lt'))
                                        greater = S.vcmp(scores_vec[i], acc, mask_32b, 'gt')
                                        acc_idx = S.vsel(index_vec[i], acc_idx, greater)
                                        acc = S.vmax(acc, scores_vec[i], mask_32b)
                                mx = S.vdupv(S.vcmax(acc, mask_32b), mask_32b)
                                acc_idx = S.vsel(acc_idx, int_max, S.vcmp(acc, mx, mask_32b, 'eq'))
                                idx = S.vdupv(S.vcmin(acc_idx, mask_32b), mask_32b)

                                # Store winner index.
                                S.vsts(out_ub[2 * k], idx, one, 'ONEPT_B32')

                                # Clear only the winner; clamp sentinels to keep the error path in bounds.
                                winner_offset = T.reinterpret(S.vmins(idx, num_experts - 1), 'uint32x64')
                                S.vscatter(neg_inf, scores_ub[0], winner_offset, one)

                                S.mem_bar('VST_VLD')

                    T.copy(out_ub[: num_topk * 2], topk_idx_view[row, :])

            T.device_assert(invalid_logits_ub[0] == 0, msg='topk_gate input contains nan or inf!')

    return topk_gate_kernel_asc

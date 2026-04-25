import math
import torch
import tilelang
from tilelang import language as T
from typing import Optional
import os

from tile_kernels.utils import align, ceil_div
from tile_kernels.moe.scoring import ScoringFunc, softplus
from tile_kernels.moe.common import get_topk_group_idx
from tile_kernels.config import get_warp_size


@T.macro
def warp_reduce_sum(x: T.Ref, warp_size: int = 32):
    n_steps = int(math.log2(warp_size))
    for i in T.unroll(0, n_steps):
        x += T.shfl_xor(x, 1 << (n_steps - 1 - i), width=warp_size)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_OUT_OF_BOUND_WARNING: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def get_top2_sum_gate_kernel(
    scoring_type: int,
    num_topk: int,
    num_topk_groups: int, num_groups: int,
    num_routed_experts: int,
    mask_exists: bool, fix_routing_mask_exists: bool,
    unmapped_topk_idx_exists: bool, to_physical_map_exists: bool,
):  # fmt: off
    # Kernel config — logical warp_size=32 for algorithmic correctness.
    # The top-k tie-breaking semantics are defined by the 5-step (offsets
    # 1,2,4,8,16) butterfly reduction with width=32. Using a wider reduction
    # (warp_size=64) changes the comparison order and produces different results
    # for equal-score experts, breaking the CUDA-compatible test contract.
    # On CDNA (wave64) the width=32 shfl calls keep shuffles within the active
    # 32-lane half of the wavefront, avoiding reads from uninitialised VGPRs.
    warp_size = 32
    num_threads = warp_size
    n_reduce_steps = int(math.log2(warp_size))
    assert num_topk <= warp_size, f'num_topk must be less than or equal to {warp_size}'

    # Each warp handles one token
    num_tokens_per_block = num_threads // warp_size

    # Keep the same with the old implementation
    large_prime_number = 23333
    num_vectorize = 4
    num_topk_sum = 2
    assert num_routed_experts % num_vectorize == 0, f'`num_routed_experts` must be divisible by {num_vectorize}, but got {num_routed_experts}'

    # Runtime symbols
    num_tokens = T.dynamic('num_tokens')
    unmapped_topk_idx_stride = T.dynamic('unmapped_topk_idx_stride')
    # num_physical_topk = num_topk + num_shared_experts
    num_physical_topk = T.dynamic('num_physical_topk')
    # num_logical_experts = num_routed_experts + num_shared_experts
    num_logical_experts = T.dynamic('num_logical_experts')
    # num_duplicate_experts = num_extra_experts + 1
    num_duplicate_experts = T.dynamic('num_duplicate_experts')

    if num_groups == num_topk_groups:
        # In this case, the value of num_groups and num_topk_groups are meaningless, so we set them to 1
        skip_group_sort = True
        num_groups = 1
        num_topk_groups = 1
    else:
        skip_group_sort = False
        assert num_groups <= warp_size, f'num_groups ({num_groups}) must be <= warp size ({warp_size}) for warp-level group ranking'
        assert num_routed_experts <= warp_size * num_groups, f'No more than {warp_size} experts per group is supported.'
        num_routed_experts_per_group = num_routed_experts // num_groups

        # Make sure that the number of routed experts is divisible by vectorization size.
        num_vectorize_for_grouped_expert = 4
        while num_routed_experts_per_group % num_vectorize_for_grouped_expert != 0:
            num_vectorize_for_grouped_expert //= 2
        assert num_routed_experts_per_group % num_vectorize_for_grouped_expert == 0

    # Number of routed experts each thread needs to handle in top-k selection
    num_routed_experts_per_thread = max(align(ceil_div(num_routed_experts, warp_size), num_vectorize), num_topk_groups)
    assert num_routed_experts_per_thread % num_vectorize == 0
    assert num_routed_experts_per_thread >= num_topk_groups

    @T.prim_func
    def top2_sum_gate_kernel(
        logits: T.Tensor[(num_tokens, num_routed_experts), T.float32],
        bias: T.Tensor[(num_routed_experts), T.float32],
        mask: T.Tensor[(num_tokens,), T.bool],
        fix_routing_mask: T.Tensor[(num_tokens,), T.bool],
        to_physical_map: T.Tensor[(num_logical_experts, num_duplicate_experts), T.int32],
        logical_count: T.Tensor[(num_logical_experts), T.int32],
        topk_idx: T.Tensor[(num_tokens, num_physical_topk), T.int64],
        unmapped_topk_idx: T.StridedTensor[(num_tokens, num_topk), (unmapped_topk_idx_stride, 1), T.int64],
        topk_weights: T.Tensor[(num_tokens, num_physical_topk), T.float32],
        num_extra_experts: T.int32,
        routed_scaling_factor: T.float32,
        ep_rank: T.int32,
        num_ep_ranks: T.int32,
        tp_rank: T.int32,
        num_tp_ranks: T.int32,
    ):
        with T.Kernel(num_tokens, threads=num_threads) as pid:
            thread_idx = T.get_thread_binding()
            token_idx = thread_idx // warp_size
            global_token_idx = token_idx + pid * num_tokens_per_block
            lane_idx = thread_idx % warp_size

            scores_shared = T.alloc_shared((num_tokens_per_block, num_routed_experts), dtype=T.float32)
            scores_wo_bias_shared = T.alloc_shared((num_tokens_per_block, num_routed_experts), dtype=T.float32)

            bias_local = T.alloc_local((num_routed_experts_per_thread,), dtype=T.float32)
            scores_local = T.alloc_local((num_routed_experts_per_thread,), dtype=T.float32)
            idx_local = T.alloc_local((num_routed_experts_per_thread,), dtype=T.int32)
            topk_group_idx_shared = T.alloc_shared((num_tokens_per_block, num_topk_groups), dtype=T.int32)
            topk_scores_local = T.alloc_local(num_topk, dtype=T.float32)
            topk_idx_local = T.alloc_local(num_topk, dtype=T.int32)
            topk_group_idx_local = T.alloc_local(num_topk_groups, dtype=T.int32)

            logit_max_var = T.alloc_var(dtype=T.float32)
            logit_sum_var = T.alloc_var(dtype=T.float32)
            other_idx = T.alloc_var(dtype=T.int32)
            topk_score_var = T.alloc_var(dtype=T.float32)
            topk_idx_var = T.alloc_var(dtype=T.int64)
            topk_sum_var = T.alloc_var(dtype=T.float32)

            # Tokens with mask = 0 does not participate in routing
            if mask_exists and not mask[global_token_idx]:
                if lane_idx < num_topk and unmapped_topk_idx_exists:
                    unmapped_topk_idx[global_token_idx, lane_idx] = -1
                if lane_idx < num_physical_topk:
                    topk_idx[global_token_idx, lane_idx] = -1
                    topk_weights[global_token_idx, lane_idx] = 0.0
                T.thread_return()

            # Load and do activation functions
            logit_max_var = -T.infinity(T.float32)
            logit_sum_var = 0.0

            # NOTES: Must be initialized before use to avoid undefined behavior
            T.fill(idx_local, -1)
            T.fill(scores_local, -T.infinity(T.float32))
            T.fill(bias_local, 0.0)

            # Load logits from global memory
            for i in T.unroll(0, num_routed_experts_per_thread // num_vectorize):
                start_expert_idx = i * num_vectorize * warp_size + lane_idx * num_vectorize
                if start_expert_idx < num_routed_experts:
                    for j in T.vectorized(num_vectorize):
                        scores_local[i * num_vectorize + j] = logits[global_token_idx, start_expert_idx + j]
                        bias_local[i * num_vectorize + j] = bias[start_expert_idx + j]
                        if scoring_type == 2:  # SOFTMAX
                            scores_shared[token_idx, start_expert_idx + j] = scores_local[i * num_vectorize + j]

                    # Test nan or inf for each element
                    for j in T.unroll(num_vectorize):
                        is_finite = T.isfinite(scores_local[i * num_vectorize + j])
                        T.device_assert(is_finite, msg='top2_sum_gate input contains nan or inf!')

            if scoring_type == 2:  # SOFTMAX
                for i in T.unroll(num_routed_experts_per_thread):
                    logit_max_var = T.max(logit_max_var, scores_local[i])
                logit_max_var = T.warp_reduce_max(logit_max_var)
                for i in T.unroll(0, T.ceildiv(num_routed_experts, num_vectorize * warp_size)):
                    if i * num_vectorize * warp_size + lane_idx * num_vectorize < num_routed_experts:
                        for j in T.unroll(num_vectorize):
                            scores_local[i * num_vectorize + j] = T.exp(scores_local[i * num_vectorize + j] - logit_max_var)
                            logit_sum_var += scores_local[i * num_vectorize + j]
                warp_reduce_sum(logit_sum_var, warp_size=warp_size)
                T.sync_warp()

            for i in T.unroll(0, T.ceildiv(num_routed_experts, num_vectorize * warp_size)):
                start_expert_idx = i * num_vectorize * warp_size + lane_idx * num_vectorize
                if start_expert_idx < num_routed_experts:
                    for j in T.vectorized(num_vectorize):
                        expert_idx = start_expert_idx + j
                        local_idx = i * num_vectorize + j
                        if scoring_type == 0:  # SIGMOID
                            scores_local[local_idx] = T.sigmoid(scores_local[local_idx])
                        elif scoring_type == 1:  # SQRTSOFTPLUS
                            scores_local[local_idx] = T.sqrt(softplus(scores_local[local_idx]))
                        elif scoring_type == 2:  # SOFTMAX
                            scores_local[local_idx] = scores_local[local_idx] / logit_sum_var
                        elif scoring_type == 3:  # IDENTITY
                            pass
                        else:
                            # Impossible branch
                            T.device_assert(0, 'Invalid scoring type')

                        scores_wo_bias_shared[token_idx, expert_idx] = scores_local[local_idx]

                        if scoring_type == 2:  # SOFTMAX
                            scores_local[local_idx] = scores_shared[token_idx, expert_idx]
                        scores_local[local_idx] += bias_local[local_idx]
                        if not skip_group_sort:
                            scores_shared[token_idx, expert_idx] = scores_local[local_idx]
                        else:
                            idx_local[local_idx] = expert_idx

            # Ensure all shared memory stores are completed
            T.sync_warp()

            if not fix_routing_mask_exists or not fix_routing_mask[global_token_idx]:
                # Get `num_topk_groups` groups with the largest top2-sum
                if not skip_group_sort:
                    # Get topk group indices
                    get_topk_group_idx(
                        scores_shared,
                        topk_group_idx_shared,
                        num_groups,
                        num_routed_experts_per_group,
                        num_topk_groups,
                        num_topk_sum,
                        num_vectorize_for_grouped_expert,
                        warp_size=warp_size,
                    )

                    # Sort group indices in ascending order to ensure stable sort
                    for i in T.vectorized(num_topk_groups):
                        topk_group_idx_local[i] = topk_group_idx_shared[token_idx, i]
                    for i in T.unroll(num_topk_groups):
                        for j in T.unroll(num_topk_groups):
                            if j > i and topk_group_idx_local[j] < topk_group_idx_local[i]:
                                swap_tmp = topk_group_idx_local[j]
                                topk_group_idx_local[j] = topk_group_idx_local[i]
                                topk_group_idx_local[i] = swap_tmp

                    # Load expert scores from shared memory
                    T.fill(scores_local, -T.infinity(T.float32))
                    T.fill(idx_local, -1)
                    if lane_idx < num_routed_experts_per_group:
                        for i in T.unroll(num_topk_groups):
                            select_group_idx = topk_group_idx_local[i]
                            scores_local[i] = scores_shared[token_idx, select_group_idx * num_routed_experts_per_group + lane_idx]
                            idx_local[i] = select_group_idx * num_routed_experts_per_group + lane_idx

                # Get topk via repeatly finding max
                for k in T.unroll(num_topk):
                    # Get local max score
                    topk_scores_local[k] = -T.infinity(T.float32)
                    for i in T.unroll(0, num_routed_experts_per_thread):
                        if k != 0 and topk_idx_local[k - 1] == idx_local[i]:
                            scores_local[i] = -T.infinity(T.float32)
                        # If j > i, then idx_local[j] > idx_local[i]
                        elif scores_local[i] > topk_scores_local[k]:
                            topk_scores_local[k] = scores_local[i]
                            topk_idx_local[k] = idx_local[i]

                    # Get max score across all threads
                    for i in T.unroll(n_reduce_steps):
                        other_score = T.shfl_xor(topk_scores_local[k], 1 << i, width=warp_size)
                        other_idx = T.shfl_xor(topk_idx_local[k], 1 << i, width=warp_size)
                        if other_score > topk_scores_local[k] or (other_score == topk_scores_local[k] and other_idx < topk_idx_local[k]):
                            topk_scores_local[k] = other_score
                            topk_idx_local[k] = other_idx

                topk_score_var = 0.0
                if lane_idx < num_topk:
                    topk_idx_var = topk_idx_local[lane_idx]
                    topk_score_var = scores_wo_bias_shared[token_idx, topk_idx_var]

            else:
                topk_score_var = 0.0
                if lane_idx < num_topk:
                    topk_idx_var = unmapped_topk_idx[global_token_idx, lane_idx]
                    topk_score_var = scores_wo_bias_shared[token_idx, topk_idx_var]

            # Get topk sum
            topk_sum_var = 1e-20
            for i in T.unroll(num_topk):
                topk_sum_var += T.shfl_sync(topk_score_var, i, width=warp_size)

            # Ensure one warp can handle one token
            T.device_assert(num_physical_topk <= warp_size)

            # Normalize top-k weights
            if lane_idx < num_topk:
                # NOTES: If this fails, there may be some NaN values in logits input or internal error in the kernel
                T.device_assert(topk_idx_var >= 0)
                topk_score_var = topk_score_var / topk_sum_var * routed_scaling_factor
                if unmapped_topk_idx_exists:
                    unmapped_topk_idx[global_token_idx, lane_idx] = topk_idx_var
            elif lane_idx < num_physical_topk:
                topk_score_var = 1.0
                topk_idx_var = lane_idx + (num_routed_experts - num_topk)

            # Map to physical experts
            if to_physical_map_exists and lane_idx < num_physical_topk:
                logical_expert_idx = topk_idx_var
                num_duplicates = logical_count[logical_expert_idx]
                duplicate_idx = (ep_rank + global_token_idx * large_prime_number) % num_duplicates
                topk_idx_var = to_physical_map[logical_expert_idx, duplicate_idx]

            # Mask ETP idx
            num_experts_per_rank = (num_routed_experts + num_extra_experts) // num_ep_ranks
            num_experts_per_dp = num_experts_per_rank * num_tp_ranks
            if lane_idx < num_physical_topk:
                dst_ep_rank = topk_idx_var // num_experts_per_rank
                if dst_ep_rank % num_tp_ranks != T.int64(tp_rank):
                    topk_idx_var = -1
                else:
                    topk_idx_var -= tp_rank * num_experts_per_rank
                    dst_dp_rank = topk_idx_var // num_experts_per_dp
                    topk_idx_var = topk_idx_var - dst_dp_rank * num_experts_per_dp + dst_dp_rank * num_experts_per_rank
                    topk_idx_var = T.if_then_else(topk_idx_var < 0, -1, topk_idx_var)
                topk_idx[global_token_idx, lane_idx] = topk_idx_var
                topk_weights[global_token_idx, lane_idx] = topk_score_var

    return top2_sum_gate_kernel


def top2_sum_gate(
    logits: torch.Tensor,
    bias: torch.Tensor,
    num_topk: int,
    num_topk_groups: int,
    num_groups: int,
    use_shared_as_routed: bool,
    num_shared_experts: int,
    routed_scaling_factor: float,
    ep_rank: int,
    num_ep_ranks: int,
    tp_rank: int,
    num_tp_ranks: int,
    scoring_func: str,
    mask: Optional[torch.Tensor] = None,
    fix_routing_mask: Optional[torch.Tensor] = None,
    to_physical_map: Optional[torch.Tensor] = None,
    logical_count: Optional[torch.Tensor] = None,
    unmapped_topk_idx: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute top-k expert routing with top-2 sum grouping for Mixture of Experts (MoE) models. Stable sort is supported.
    Args:
        logits: Input token-expert scores of shape (num_tokens, num_routed_experts)
        bias: Expert bias terms of shape (num_routed_experts,)
        num_topk: Number of top experts to select per token
        num_topk_groups: Number of top expert groups to consider
        num_groups: Total number of expert groups
        use_shared_as_routed: Whether to treat shared experts as routed experts
        num_shared_experts: Number of shared experts
        routed_scaling_factor: Scaling factor for routed expert weights
        ep_rank: Current expert parallelism rank
        num_ep_ranks: Total number of expert parallelism ranks
        tp_rank: Current tensor parallelism rank
        num_tp_ranks: Total number of tensor parallelism ranks
        scoring_func: Scoring function type - 'sigmoid', 'sqrtsoftplus', 'softmax'
        mask: Optional mask to skip routing for specific tokens (True = route, False = skip)
        fix_routing_mask: Optional mask for fixed routing
        to_physical_map: Optional mapping from logical to physical experts
        logical_count: Optional count of logical experts
        unmapped_topk_idx: Optional output tensor for unmapped expert indices

    Returns:
        tuple:
        - topk_idx: Selected expert indices of shape (num_tokens, num_physical_topk)
        - topk_weights: Corresponding expert weights of shape (num_tokens, num_physical_topk)
    """
    assert logits.dim() == 2 and logits.is_contiguous() and logits.dtype == torch.float32
    assert bias.dim() == 1 and bias.is_contiguous() and bias.dtype == torch.float32
    assert logits.size(1) == bias.size(0)
    if mask is not None:
        assert mask.numel() == logits.size(0) and mask.dtype == torch.bool
    assert (to_physical_map is None) == (logical_count is None)

    num_tokens = logits.size(0)
    num_routed_experts = logits.size(1)
    assert (num_groups == 0) == (num_topk_groups == 0), 'num_groups and num_topk_groups must be both zero or both non-zero.'
    assert num_topk <= num_routed_experts, f'num_topk ({num_topk}) must be less than or equal to num_routed_experts ({num_routed_experts}).'
    assert num_topk_groups <= num_groups, f'num_topk_groups ({num_topk_groups}) must be less than or equal to num_groups ({num_groups}).'
    if num_groups != 0:
        assert num_routed_experts % num_groups == 0, f'num_routed_experts ({num_routed_experts}) must be divisible by num_groups ({num_groups}).'

    assert ScoringFunc.from_str(scoring_func) != ScoringFunc.IDENTITY, 'IDENTITY scoring function is not currently supported for top-2 sum moe.'

    if not use_shared_as_routed:
        num_shared_experts = 0
    else:
        assert num_topk % num_shared_experts == 0
        assert num_routed_experts % (num_topk // num_shared_experts) == 0

    assert num_shared_experts in {1, 2} or (num_shared_experts == 0 and not use_shared_as_routed)

    topk_idx = torch.empty(num_tokens, num_topk + num_shared_experts, dtype=torch.long, layout=logits.layout, device=logits.device)
    topk_weights = torch.empty(num_tokens, num_topk + num_shared_experts, dtype=torch.float32, layout=logits.layout, device=logits.device)
    if num_tokens == 0:
        return topk_idx, topk_weights

    num_extra_experts = 0
    num_logical_experts = num_shared_experts + num_routed_experts
    if to_physical_map is not None:
        assert to_physical_map.is_contiguous()
        assert to_physical_map.dim() == 2 and to_physical_map.dtype == torch.int32
        assert to_physical_map.size(0) == num_logical_experts
        num_extra_experts = to_physical_map.size(1) - 1
        assert logical_count is not None and logical_count.is_contiguous()
        assert logical_count.dim() == 1 and logical_count.dtype == torch.int32
        assert logical_count.size(0) == num_logical_experts

    assert num_shared_experts <= num_extra_experts

    if unmapped_topk_idx is not None:
        assert unmapped_topk_idx.dim() == 2 and unmapped_topk_idx.dtype == torch.long
        assert unmapped_topk_idx.size(0) == num_tokens and unmapped_topk_idx.size(1) == num_topk
        assert unmapped_topk_idx.stride(1) == 1

    if fix_routing_mask is not None:
        assert unmapped_topk_idx is not None
        assert fix_routing_mask.is_contiguous()
        assert fix_routing_mask.dtype == torch.bool
        assert fix_routing_mask.dim() == 1 and fix_routing_mask.size(0) == num_tokens

    kernel = get_top2_sum_gate_kernel(
        ScoringFunc.from_str(scoring_func).value,
        num_topk,
        num_topk_groups, num_groups,
        num_routed_experts,
        mask is not None, fix_routing_mask is not None,
        unmapped_topk_idx is not None, to_physical_map is not None,
    )  # fmt: off

    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    kernel(logits, bias,
           mask, fix_routing_mask, to_physical_map, logical_count,
           topk_idx, unmapped_topk_idx, topk_weights,
           num_extra_experts, routed_scaling_factor,
           ep_rank, num_ep_ranks, tp_rank, num_tp_ranks)  # fmt: off

    return topk_idx, topk_weights

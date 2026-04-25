import os
import torch
import tilelang
from tilelang import language as T
from tile_kernels.config import get_num_sms, get_warp_size

from tile_kernels.utils import align


@T.macro
def divide_task(length: int, num_tasks: int, task_id: int, start: T.Ref, end: T.Ref):
    length_per_task = align(T.ceildiv(length, num_tasks), 32)
    start = task_id * length_per_task
    end = T.min(start + length_per_task, length)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_DISABLE_OUT_OF_BOUND_WARNING: True,
    },
)
def get_get_fused_mapping_kernel(
    num_experts: int,
    num_topk: int,
    alignment: int,
    num_sms: int,
):
    num_threads = 256
    while num_threads < num_experts:
        num_threads *= 2
    assert num_threads <= 1024 and num_threads >= num_experts
    warp_size = get_warp_size()
    num_warps = num_threads // warp_size

    num_global_warps = num_sms * num_warps
    num_global_threads = num_threads * num_sms
    # Runtime symbols
    num_tokens = T.dynamic('num_tokens')
    num_expanded_tokens = T.dynamic('num_expanded_tokens')

    @T.prim_func
    def get_fused_mapping_kernel(
        topk_idx: T.Tensor[(num_tokens, num_topk), T.int64],
        pos_to_expert: T.Tensor[(num_expanded_tokens,), T.int32],
        pos_to_token: T.Tensor[(num_expanded_tokens,), T.int32],
        pos_to_token_topk: T.Tensor[(num_expanded_tokens,), T.int32],
        token_topk_to_pos: T.Tensor[(num_tokens, num_topk), T.int32],
        expert_start: T.Tensor[(num_experts,), T.int32],
        expert_end: T.Tensor[(num_experts,), T.int32],
        num_tokens_per_expert: T.Tensor[(num_experts,), T.int32],
        num_experts_per_sm: T.Tensor[(num_sms, num_experts), T.int32],
    ):
        with T.Kernel(num_sms, threads=num_threads) as (sm_idx,):
            thread_idx = T.get_thread_binding(0)
            warp_idx = thread_idx // warp_size
            lane_idx = thread_idx % warp_size
            global_thread_idx = sm_idx * num_threads + thread_idx
            global_warp_idx = sm_idx * num_warps + warp_idx
            numel = num_tokens * num_topk
            experts_sum_per_warp_shared = T.alloc_shared((num_warps, num_experts), T.int32)
            # The number of elements per expert
            num_elems_per_expert_shared = T.alloc_shared((num_experts,), T.int32)
            # The number of elements per expert processed by SMs 0 to sm_idx − 1
            num_elems_prefix_shared = T.alloc_shared((num_experts,), T.int32)

            T.clear(num_elems_per_expert_shared)
            T.clear(num_elems_prefix_shared)

            topk_idx_1d = T.view(topk_idx, (num_tokens * num_topk,))
            token_topk_to_pos_1d = T.view(token_topk_to_pos, (num_tokens * num_topk,))

            for i in T.serial(lane_idx, num_experts, warp_size):
                experts_sum_per_warp_shared[warp_idx, i] = 0
            T.sync_warp()

            for i in T.serial(global_thread_idx, num_expanded_tokens, num_global_threads):
                pos_to_token[i] = -1
                pos_to_token_topk[i] = -1
                pos_to_expert[i] = -1

            for i in T.serial(global_thread_idx, numel, num_global_threads):
                token_topk_to_pos_1d[i] = -1

            start = T.alloc_var(T.int32)
            end = T.alloc_var(T.int32)
            divide_task(numel, num_global_warps, global_warp_idx, start, end)

            for i in T.serial(start + lane_idx, end, warp_size):
                T.assume(0 <= i < numel)
                expert_idx = topk_idx_1d[i]
                if expert_idx != -1:
                    T.assume(0 <= expert_idx < num_experts)
                    T.atomic_add(experts_sum_per_warp_shared[warp_idx, expert_idx], 1)

            T.sync_threads()

            if thread_idx < num_experts:
                for i in T.unroll(num_warps - 1):
                    experts_sum_per_warp_shared[i + 1, thread_idx] += experts_sum_per_warp_shared[i, thread_idx]
                num_experts_per_sm[sm_idx, thread_idx] = experts_sum_per_warp_shared[num_warps - 1, thread_idx]

            T.sync_grid()

            cumsum_shared = T.alloc_shared((num_threads,), T.int32)
            expert_num_elements = T.alloc_var(T.int32, init=0)
            expert_num_elements_aligned = T.alloc_var(T.int32, init=0)
            prefix_expert_num_elements = T.alloc_var(T.int32, init=0)

            # Obtain and align the number of elements for the expert corresponding to thread_id,
            # compute the prefix sum, and then determine the start and end positions for each expert.
            if thread_idx < num_experts:
                for i in T.serial(0, num_sms):
                    T.assume(i < num_sms * num_experts)
                    num = num_experts_per_sm[i, thread_idx]
                    expert_num_elements += num
                    if i < sm_idx:
                        prefix_expert_num_elements += num
                expert_num_elements_aligned = align(expert_num_elements, alignment)
                cumsum_shared[thread_idx] = expert_num_elements_aligned

            T.sync_threads()
            T.cumsum(cumsum_shared)
            T.sync_threads()
            exclusive_prefix = cumsum_shared[thread_idx] - expert_num_elements_aligned

            if thread_idx < num_experts:
                # Apply expert_prefix_shared to the warp prefix sum
                for i in T.unroll(num_warps):
                    experts_sum_per_warp_shared[i, thread_idx] += prefix_expert_num_elements + exclusive_prefix

                # Write the start and end positions of each expert to global memory
                if sm_idx == 0:
                    num_tokens_per_expert[thread_idx] = expert_num_elements_aligned
                    expert_start[thread_idx] = exclusive_prefix
                    expert_end[thread_idx] = exclusive_prefix + expert_num_elements_aligned
            T.sync_threads()

            divide_task(numel, num_global_warps, global_warp_idx, start, end)
            aligned_end = align(end, warp_size)
            lane_mask = T.uint32(1 << lane_idx) + T.uint32(1 << lane_idx) - 1
            lane_mask_rev = ~lane_mask
            for i in T.serial(start + lane_idx, aligned_end, warp_size):
                T.assume(0 <= i)
                expert_idx = T.Select(i < numel, T.int32(topk_idx_1d[i]), -1)
                mask = T.call_extern(T.uint32, '__match_any_sync', 0xFFFFFFFF, expert_idx)
                count = T.popcount(mask & lane_mask)

                if i < numel and expert_idx >= 0:
                    T.assume(expert_idx < num_experts)
                    prefix_count = experts_sum_per_warp_shared[warp_idx, expert_idx]
                    pos = prefix_count - count
                    if mask & lane_mask_rev == 0:
                        experts_sum_per_warp_shared[warp_idx, expert_idx] = pos
                    token_topk_to_pos_1d[i] = pos
                    T.assume(0 <= pos < num_expanded_tokens)
                    pos_to_expert[pos] = expert_idx
                    pos_to_token[pos] = i // num_topk
                    pos_to_token_topk[pos] = i
                T.sync_warp()

    return get_fused_mapping_kernel


def get_fused_mapping(
    topk_idx: torch.Tensor,
    num_experts: int,
    num_expanded_tokens: int,
    alignment: int,
    force_no_sync: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Build fused MoE routing mappings between token-topk and expert-major layouts.

    Args:
        topk_idx: Int64 expert index tensor of shape (num_tokens, num_topk).
        num_experts: Total number of experts.
        num_expanded_tokens: Size of fused expert-major buffer. If set to ``0`` and
            ``force_no_sync`` is ``False``, it is estimated and then trimmed using
            ``num_tokens_per_expert`` with a host sync.
        alignment: Per-expert alignment size used for fused layout packing.
        force_no_sync: Whether to skip host-side synchronization when
            ``num_expanded_tokens`` is ``0``.

    Returns:
        A tuple ``(
            pos_to_expert,
            pos_to_token,
            pos_to_token_topk,
            token_topk_to_pos,
            expert_start,
            expert_end,
            num_tokens_per_expert,
            num_tokens_per_expert_list,
        )`` containing fused layout mappings, per-expert ranges, and optional
        synchronized per-expert counts.
    """
    num_tokens, num_topk = topk_idx.shape
    assert topk_idx.is_contiguous() and topk_idx.dtype == torch.int64

    should_sync = False
    if num_expanded_tokens == 0 and not force_no_sync:
        should_sync = True
        num_expanded_tokens = (num_tokens * num_topk + (alignment - 1) * num_experts) // alignment * alignment

    # Allocate output
    num_sms = get_num_sms()
    pos_to_expert = torch.empty((num_expanded_tokens, ), dtype=torch.int32, device='cuda')
    pos_to_token = torch.empty((num_expanded_tokens, ), dtype=torch.int32, device='cuda')
    pos_to_token_topk = torch.empty((num_expanded_tokens, ), dtype=torch.int32, device='cuda')
    token_topk_to_pos = torch.empty((num_tokens, num_topk), dtype=torch.int32, device='cuda')
    expert_start = torch.empty((num_experts, ), dtype=torch.int32, device='cuda')
    expert_end = torch.empty((num_experts, ), dtype=torch.int32, device='cuda')
    num_tokens_per_expert = torch.empty((num_experts, ), dtype=torch.int32, device='cuda')
    num_experts_per_sm = torch.empty((num_sms, num_experts), dtype=torch.int32, device='cuda')

    # Get kernel and launch
    mapping_kernel = get_get_fused_mapping_kernel(num_experts, num_topk, alignment, num_sms)
    mapping_kernel(
        topk_idx,
        pos_to_expert,
        pos_to_token,
        pos_to_token_topk,
        token_topk_to_pos,
        expert_start,
        expert_end,
        num_tokens_per_expert,
        num_experts_per_sm,
    )
    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(mapping_kernel.get_kernel_source())

    # May involve CPU sync
    num_tokens_per_expert_list = []
    if should_sync:
        num_tokens_per_expert_list = num_tokens_per_expert.tolist()
        num_expanded_tokens = sum(num_tokens_per_expert_list)
        pos_to_expert = pos_to_expert[:num_expanded_tokens]
        pos_to_token = pos_to_token[:num_expanded_tokens]
        pos_to_token_topk = pos_to_token_topk[:num_expanded_tokens]
    return (
        pos_to_expert,
        pos_to_token,
        pos_to_token_topk,
        token_topk_to_pos,
        expert_start,
        expert_end,
        num_tokens_per_expert,
        num_tokens_per_expert_list,
    )

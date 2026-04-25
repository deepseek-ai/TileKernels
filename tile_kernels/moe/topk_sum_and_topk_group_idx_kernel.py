import os

import tilelang
import torch
from tilelang import language as T

from tile_kernels.moe.common import get_topk_group_idx
from tile_kernels.utils import align
from tile_kernels.config import get_warp_size


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_OUT_OF_BOUND_WARNING: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def get_topk_sum_and_topk_group_idx_kernel(
    num_groups: int,
    num_experts_per_group: int,
    num_topk_groups: int,
    num_topk_sum: int,
):
    warp_size = get_warp_size()
    num_threads = warp_size
    num_experts = num_experts_per_group * num_groups
    num_aligned_experts = align(num_experts, num_threads)
    num_tokens_per_block = 1  # one wavefront per token

    assert num_groups <= warp_size, f'num_groups ({num_groups}) must be <= warp size ({warp_size})'

    # Make sure that the number of experts per group is divisible by vectorization size.
    num_vectorize_for_grouped_expert = 4
    while num_experts_per_group % num_vectorize_for_grouped_expert != 0:
        num_vectorize_for_grouped_expert //= 2
    assert num_experts_per_group % num_vectorize_for_grouped_expert == 0

    num_tokens = T.dynamic('num_tokens')

    @T.prim_func
    def topk_sum_and_topk_group_idx_kernel(
        scores: T.Tensor[(num_tokens, num_experts), T.float32],
        group_topk_idx: T.Tensor[(num_tokens, num_topk_groups), T.int64],
    ):
        with T.Kernel(num_tokens, threads=num_threads) as pid:
            scores_shared = T.alloc_shared((num_tokens_per_block, num_aligned_experts), T.float32)
            topk_group_idx_shared = T.alloc_shared((num_tokens_per_block, num_topk_groups), T.int32)

            thread_idx = T.get_thread_binding()
            warp_idx = thread_idx // warp_size
            lane_idx = thread_idx % warp_size

            T.copy(scores[pid * num_tokens_per_block, 0], scores_shared)
            T.sync_warp()

            get_topk_group_idx(
                scores_shared=scores_shared,
                topk_group_idx_shared=topk_group_idx_shared,
                num_groups=num_groups,
                num_experts_per_group=num_experts_per_group,
                num_topk_groups=num_topk_groups,
                num_topk_sum=num_topk_sum,
                num_vectorize_for_grouped_expert=num_vectorize_for_grouped_expert,
                warp_size=warp_size,
            )

            if lane_idx < num_topk_groups:
                group_topk_idx[pid * num_tokens_per_block + warp_idx, lane_idx] = topk_group_idx_shared[warp_idx, lane_idx]

    return topk_sum_and_topk_group_idx_kernel


def topk_sum_and_topk_group_idx(scores: torch.Tensor, num_topk_sum: int, num_topk_groups: int) -> torch.Tensor:
    """Return top-``num_topk_groups`` group indices ranked by intra-group top-k sum.

    For each token, this function computes a group score by summing the largest
    ``num_topk_sum`` expert scores within each group, then returns the indices of
    the groups with the highest summed values.

    Args:
        scores: Contiguous float32 tensor with shape
            ``(num_tokens, num_groups, num_experts_per_group)``.
        num_topk_sum: Number of top expert scores to sum per group. Only ``1``
            and ``2`` are supported.
        num_topk_groups: Number of highest-scoring groups to select per token.

    Returns:
        ``torch.int64`` tensor of shape ``(num_tokens, num_topk_groups)``
        containing selected group indices for each token.
    """
    assert scores.dim() == 3 and scores.is_contiguous() and scores.dtype == torch.float32
    num_tokens, num_groups, num_experts_per_group = scores.shape
    assert num_topk_sum <= num_experts_per_group and num_topk_sum in (1, 2) and num_topk_groups <= num_groups

    kernel = get_topk_sum_and_topk_group_idx_kernel(num_groups, num_experts_per_group, num_topk_groups, num_topk_sum)
    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    topk_group_idx = torch.empty(num_tokens, num_topk_groups, dtype=torch.int64, device=scores.device)
    if num_tokens == 0:
        return topk_group_idx

    kernel(scores.view(num_tokens, -1), topk_group_idx)
    return topk_group_idx

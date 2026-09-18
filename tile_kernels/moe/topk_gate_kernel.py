import os

import tilelang
import torch
from tilelang import language as T

from tile_kernels.utils import align, get_device_guard


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_topk_gate_kernel(num_experts: int, num_topk: int):
    num_tokens = T.dynamic('num_tokens')
    num_threads = 32
    num_aligned_experts = align(num_experts, num_threads)

    @T.prim_func
    def topk_gate_kernel(
        scores: T.Tensor[(num_tokens, num_experts), T.float32],
        topk_idx: T.Tensor[(num_tokens, num_topk), T.int64],
    ):
        with T.Kernel(num_tokens, threads=num_threads) as pid:
            scores_fragment = T.alloc_fragment((num_aligned_experts,), T.float32)
            amax_fragment = T.alloc_fragment((1,), T.float32)
            idx_fragment = T.alloc_fragment((num_aligned_experts,), T.int32)
            idx_reducer = T.alloc_reducer((1,), T.int32, 'min', replication='all')
            topk_idx_shared = T.alloc_shared((num_topk,), T.int32)

            for i in T.Parallel(num_aligned_experts):
                if i < num_experts:
                    scores_fragment[i] = scores[pid, i]
                else:
                    scores_fragment[i] = -T.infinity(T.float32)
            for i in T.Parallel(num_aligned_experts):
                idx_fragment[i] = i

            # Get topk via repeatly finding max
            for k in T.unroll(num_topk):
                T.reduce_max(scores_fragment, amax_fragment)
                T.fill(idx_reducer, T.max_value(T.int32))
                for i in T.Parallel(num_aligned_experts):
                    if scores_fragment[i] == amax_fragment[0]:
                        idx_reducer[0] = T.min(idx_reducer[0], idx_fragment[i])
                T.finalize_reducer(idx_reducer)
                topk_idx_shared[k] = idx_reducer[0]
                for i in T.Parallel(num_aligned_experts):
                    if idx_fragment[i] == idx_reducer[0]:
                        scores_fragment[i] = -T.infinity(T.float32)

            T.copy(topk_idx_shared, topk_idx[pid, 0], disable_tma=True)

    return topk_gate_kernel


def topk_gate(scores: torch.Tensor, num_topk: int) -> torch.Tensor:
    """Select the top-k experts per token from scores.

    Args:
        scores (torch.Tensor): Gating logits or scores with shape
            ``[num_tokens, num_experts]``. Higher values indicate stronger
            routing preference.
        num_topk (int): Number of experts to select per token. Must satisfy
            ``1 <= num_topk <= num_experts``.

    Returns:
        torch.Tensor: Top-k expert indices with shape ``[num_tokens, num_topk]``
            and ``torch.int64``. Each row contains the selected
            expert indices for the corresponding token.

    Notes:
        - Always return the smaller index when there are ties.
        - The output is always contiguous.
    """
    assert scores.dim() == 2 and scores.is_contiguous() and scores.dtype == torch.float32
    num_tokens, num_experts = scores.shape
    assert num_topk <= num_experts, f'num_topk ({num_topk}) must be <= num_experts ({num_experts})'
    topk_idx = torch.empty((num_tokens, num_topk), dtype=torch.int64, device=scores.device)
    if num_tokens == 0:
        return topk_idx

    # Compile and launch on the device of the input tensor, which is not
    # necessarily the current CUDA device.
    with get_device_guard(scores.device):
        kernel = get_topk_gate_kernel(num_experts, num_topk)

        if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
            print(kernel.get_kernel_source())

        kernel(scores, topk_idx)
    return topk_idx

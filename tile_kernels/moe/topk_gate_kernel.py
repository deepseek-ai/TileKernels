import torch

from tile_kernels.config import get_num_vec_cores, get_pdl, is_ascend
from tile_kernels.moe.topk_gate_cuda import get_topk_gate_kernel_cuda
from tile_kernels.utils import get_device_guard


def topk_gate(scores: torch.Tensor, num_topk: int) -> torch.Tensor:
    """Select the top-k experts per token from scores.

    Args:
        scores (torch.Tensor): Gating logits or scores with shape
            ``[num_tokens, num_experts]``. Higher values indicate stronger
            routing preference.
        num_topk (int): Number of experts to select per token. Must satisfy
            ``1 <= num_topk <= min(num_experts, 32)``.

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
    assert num_topk <= 32, f'num_topk ({num_topk}) must be <= 32'

    topk_idx = torch.empty((num_tokens, num_topk), dtype=torch.int64, device=scores.device)
    if num_tokens == 0:
        return topk_idx

    # The input tensor decides which GPU this runs on; the current device must not.
    device_index = scores.device.index
    with get_device_guard(scores.device):
        if is_ascend():
            from tile_kernels.moe.topk_gate_asc import get_topk_gate_kernel_asc

            kernel = get_topk_gate_kernel_asc(num_experts, num_topk, get_num_vec_cores(device_index))
        else:
            kernel = get_topk_gate_kernel_cuda(num_experts, num_topk, use_pdl=get_pdl())

        kernel(scores, topk_idx)

    return topk_idx

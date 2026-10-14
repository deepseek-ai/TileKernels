import torch

from tile_kernels.config import get_num_vec_cores, get_pdl, is_ascend
from tile_kernels.utils import get_device_guard

from .normalize_weight_cuda import get_normalize_weight_kernel_cuda


def normalize_weight(topk_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize top-k routing weights so that each token's weights sum to one.

    Args:
        topk_weights: FP32 tensor of shape (num_tokens, num_topk), with num_topk <= 32.

    Returns:
        A tuple ``(denominator, normalized_weights)`` where ``denominator`` has
        shape (num_tokens,) and ``normalized_weights`` has shape (num_tokens, num_topk).
    """
    assert topk_weights.dim() == 2 and topk_weights.is_contiguous()
    assert topk_weights.dtype == torch.float32

    num_tokens, num_topk = topk_weights.shape
    assert num_topk <= 32, f'num_topk ({num_topk}) must be <= 32'

    # The input tensor decides which GPU this runs on; the current device must not.
    device_index = topk_weights.device.index
    with get_device_guard(topk_weights.device):
        if is_ascend():
            from .normalize_weight_asc import get_normalize_weight_kernel_asc

            kernel = get_normalize_weight_kernel_asc(num_topk=num_topk, num_vec_cores=get_num_vec_cores(device_index))
        else:
            kernel = get_normalize_weight_kernel_cuda(num_topk=num_topk, use_pdl=get_pdl())

        denominator = torch.empty((num_tokens,), dtype=torch.float32, device=topk_weights.device)
        normalized_weights = torch.empty((num_tokens, num_topk), dtype=torch.float32, device=topk_weights.device)

        if num_tokens > 0:
            kernel(topk_weights, denominator, normalized_weights)

    return (denominator, normalized_weights)

import torch
from tilelang import language as T

from tile_kernels.config import get_num_vec_cores, get_pdl, is_ascend
from tile_kernels.engram.engram_fused_weight_cuda import get_engram_fused_weight_kernel_cuda


def fused_weight(weight_hidden: torch.Tensor, weight_embed: torch.Tensor) -> torch.Tensor:
    """Compute weight_hidden * weight_embed in fp32.

    Args:
        weight_hidden: Shape (hc_mult, hidden_size), bfloat16 or float32.
        weight_embed: Shape (hc_mult, hidden_size), same dtype as weight_hidden.

    Returns:
        weight_fused: Shape (hc_mult, hidden_size), float32.
    """
    hc_mult, hidden_size = weight_hidden.shape

    if is_ascend():
        from tile_kernels.engram.engram_fused_weight_asc import get_engram_fused_weight_kernel_asc

        kernel = get_engram_fused_weight_kernel_asc(
            hidden_size,
            hc_mult,
            dtype=T.dtype(weight_hidden.dtype),
            num_vec_cores=get_num_vec_cores(),
        )
    else:
        kernel = get_engram_fused_weight_kernel_cuda(
            hidden_size,
            hc_mult,
            dtype=T.dtype(weight_hidden.dtype),
            use_pdl=get_pdl(),
        )

    weight_fused = torch.empty(hc_mult, hidden_size, dtype=torch.float32, device=weight_hidden.device)
    kernel(weight_hidden, weight_embed, weight_fused)

    return weight_fused

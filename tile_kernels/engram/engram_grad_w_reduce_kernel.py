import torch
from tilelang import language as T

from tile_kernels.config import get_num_vec_cores, get_pdl, is_ascend
from tile_kernels.engram.engram_grad_w_reduce_cuda import (
    get_engram_grad_w_reduce_kernel_cuda,
)


def grad_w_reduce(
    grad_w_partial: torch.Tensor,
    weight_hidden: torch.Tensor,
    weight_embed: torch.Tensor,
    grad_weight_hidden: torch.Tensor,
    grad_weight_embed: torch.Tensor,
    grad_weight_value: torch.Tensor | None = None,
) -> None:
    """Reduce grad_w_partial over persistent blocks, fused with weight multiply, accumulating into grad_weight tensors.

    Args:
        grad_w_partial: Partial weight gradients, shape (num_persistent_blocks, hc_mult + has_value_norm, hidden_size), float32.
        weight_hidden: RMSNorm weight for hidden states, shape (hc_mult, hidden_size), bfloat16 or float32.
        weight_embed: RMSNorm weight for key embeddings, shape (hc_mult, hidden_size), same dtype as weight_hidden.
        grad_weight_hidden: Accumulated gradient for weight_hidden, shape (hc_mult, hidden_size), float32. Modified in-place.
        grad_weight_embed: Accumulated gradient for weight_embed, shape (hc_mult, hidden_size), float32. Modified in-place.
        grad_weight_value: Optional accumulated gradient for weight_value, shape (hidden_size,), float32. Modified in-place.
    """
    num_persistent_blocks = grad_w_partial.shape[0]
    hc_mult, hidden_size = weight_hidden.shape
    has_value_norm = grad_weight_value is not None

    if is_ascend():
        from tile_kernels.engram.engram_grad_w_reduce_asc import (
            get_engram_grad_w_reduce_kernel_asc,
        )

        kernel = get_engram_grad_w_reduce_kernel_asc(
            hidden_size,
            num_persistent_blocks,
            hc_mult,
            has_value_norm=has_value_norm,
            dtype=T.dtype(weight_hidden.dtype),
            num_vec_cores=get_num_vec_cores(),
        )
    else:
        kernel = get_engram_grad_w_reduce_kernel_cuda(
            hidden_size,
            num_persistent_blocks,
            hc_mult,
            has_value_norm=has_value_norm,
            dtype=T.dtype(weight_hidden.dtype),
            use_pdl=get_pdl(),
        )

    kernel(grad_w_partial, weight_hidden, weight_embed, grad_weight_hidden, grad_weight_embed, grad_weight_value)

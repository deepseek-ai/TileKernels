from typing import Optional

import torch

from tile_kernels.config import get_num_vec_cores, get_pdl, is_ascend
from tile_kernels.moe.moe_topk_gate_forward_cuda import get_moe_topk_gate_forward_kernel_cuda
from tile_kernels.utils import get_device_guard


def moe_topk_gate_forward(
    logits: torch.Tensor,
    num_topk: int,
    use_shared_as_routed: bool,
    num_shared_experts: int,
    routed_scaling_factor: float,
    ep_rank: int,
    scoring_func: str = 'sqrtsoftplus',
    mask: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    image_bias: Optional[torch.Tensor] = None,
    image_token_mask: Optional[torch.Tensor] = None,
    fix_routing_mask: Optional[torch.Tensor] = None,
    to_physical_map: Optional[torch.Tensor] = None,
    logical_count: Optional[torch.Tensor] = None,
    unmapped_topk_idx: Optional[torch.Tensor] = None,
    force_random: Optional[torch.Tensor] = None,
    out: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    scores: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute top-k expert routing for Mixture of Experts (MoE) models. Stable sort is supported.
    Args:
        logits: Input token-expert scores of shape (num_tokens, num_routed_experts)
        num_topk: Number of top experts to select per token
        use_shared_as_routed: Whether to treat shared experts as routed experts
        num_shared_experts: Number of shared experts
        routed_scaling_factor: Scaling factor for routed expert weights
        ep_rank: Current expert parallelism rank
        scoring_func: Must be 'sqrtsoftplus' (scores are `sqrt(softplus(logits))`); kept in the signature for the callers' config plumbing
        mask: Optional mask to skip routing for specific tokens (True = route, False = skip)
        bias: Optional expert bias terms of shape (num_routed_experts,)
        image_bias: Optional expert bias terms for VL of shape (num_routed_experts,)
        image_token_mask: Optional, 'True' stands for image tokens (num_tokens,)
        fix_routing_mask: Optional mask for fixed routing
        to_physical_map: Optional mapping from logical to physical experts
        logical_count: Optional count of logical experts
        unmapped_topk_idx: Optional output tensor for unmapped expert indices
        force_random: Optional, 'True' stands for random topk idx, weights and all -1 unmapped topk idx (num_tokens,), has priority higher than fix_routing_mask but lower than mask
        out: Optional tuple of pre-allocated (topk_idx, topk_weights) tensors

    Returns:
        tuple:
        - topk_idx: Selected expert indices of shape (num_tokens, num_physical_topk)
        - topk_weights: Corresponding expert weights of shape (num_tokens, num_physical_topk)
    """
    assert scoring_func == 'sqrtsoftplus', f'moe_topk_gate_forward only supports sqrtsoftplus scoring, got {scoring_func!r}'
    assert logits.dim() == 2 and logits.is_contiguous() and logits.dtype == torch.float32
    bias_exists = bias is not None
    if bias_exists:
        assert bias.dim() == 1 and bias.is_contiguous() and bias.dtype == torch.float32
        assert logits.size(1) == bias.size(0)

    image_token_mask_exists = image_token_mask is not None
    assert image_token_mask_exists == (image_bias is not None), 'image_bias and image_token_mask must be both provided or both None'
    if image_token_mask_exists:
        assert image_bias.dim() == 1 and image_bias.is_contiguous() and image_bias.dtype == torch.float32
        assert image_token_mask.dim() == 1 and image_token_mask.is_contiguous() and image_token_mask.dtype == torch.bool
        assert logits.size(1) == image_bias.size(0)
        assert logits.size(0) == image_token_mask.size(0)

    if mask is not None:
        assert mask.numel() == logits.size(0) and mask.dtype == torch.bool
    assert (to_physical_map is None) == (logical_count is None)

    num_tokens = logits.size(0)
    num_routed_experts = logits.size(1)
    assert num_topk <= num_routed_experts, f'num_topk ({num_topk}) must be less than or equal to num_routed_experts ({num_routed_experts}).'

    if not use_shared_as_routed:
        num_shared_experts = 0

    assert num_shared_experts in {1, 2} or (num_shared_experts == 0 and not use_shared_as_routed)

    if out is None:
        topk_idx = torch.empty(num_tokens, num_topk + num_shared_experts, dtype=torch.int64, layout=logits.layout, device=logits.device)
        topk_weights = torch.empty(num_tokens, num_topk + num_shared_experts, dtype=torch.float32, layout=logits.layout, device=logits.device)
    else:
        topk_idx, topk_weights = out
        assert topk_idx.size() == (num_tokens, num_topk + num_shared_experts) and topk_idx.dtype == torch.int64
        assert topk_weights.size() == (num_tokens, num_topk + num_shared_experts) and topk_weights.dtype == torch.float32

    if scores is not None:
        assert scores.size() == logits.size() and scores.dtype == torch.float32
        assert scores.is_contiguous() and scores.device == logits.device
    if num_tokens == 0:
        return topk_idx, topk_weights

    num_logical_experts = num_shared_experts + num_routed_experts
    if to_physical_map is not None:
        assert to_physical_map.is_contiguous()
        assert to_physical_map.dim() == 2 and to_physical_map.dtype == torch.int32
        assert to_physical_map.size(0) == num_logical_experts
        assert logical_count is not None and logical_count.is_contiguous()
        assert logical_count.dim() == 1 and logical_count.dtype == torch.int32
        assert logical_count.size(0) == num_logical_experts

    if unmapped_topk_idx is not None:
        assert unmapped_topk_idx.dim() == 2 and unmapped_topk_idx.dtype == torch.int64
        assert unmapped_topk_idx.size(0) == num_tokens and unmapped_topk_idx.size(1) == num_topk
        assert unmapped_topk_idx.stride(1) == 1

    if fix_routing_mask is not None:
        assert unmapped_topk_idx is not None
        assert fix_routing_mask.is_contiguous()
        assert fix_routing_mask.dtype == torch.bool
        assert fix_routing_mask.dim() == 1 and fix_routing_mask.size(0) == num_tokens

    # The input tensor decides which GPU this runs on; the current device must not.
    device_index = logits.device.index
    with get_device_guard(logits.device):
        if is_ascend():
            from tile_kernels.moe.moe_topk_gate_forward_asc import get_moe_topk_gate_forward_kernel_asc

            kernel = get_moe_topk_gate_forward_kernel_asc(
                num_topk,
                num_routed_experts,
                num_shared_experts,
                to_physical_map.size(1) if to_physical_map is not None else 1,
                mask is not None, fix_routing_mask is not None,
                unmapped_topk_idx is not None, to_physical_map is not None,
                bias_exists, image_token_mask_exists,
                force_random is not None, scores is not None,
                get_num_vec_cores(device_index),
            )  # fmt: off
        else:
            kernel = get_moe_topk_gate_forward_kernel_cuda(
                num_topk,
                num_routed_experts,
                mask is not None, fix_routing_mask is not None,
                unmapped_topk_idx is not None, to_physical_map is not None,
                bias_exists, image_token_mask_exists,
                force_random is not None, scores is not None,
                use_pdl=get_pdl(),
            )  # fmt: off

        kernel(
            logits, bias,
            image_bias, image_token_mask,
            mask, fix_routing_mask, to_physical_map, logical_count,
            topk_idx, unmapped_topk_idx, topk_weights, scores,
            force_random,
            num_shared_experts, routed_scaling_factor,
            ep_rank
        )  # fmt: off

    return topk_idx, topk_weights


def moe_topk_gate(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility entry point for callers using the original forward name."""
    return moe_topk_gate_forward(*args, **kwargs)

from typing import Optional

import torch

from tile_kernels.config import get_pdl, is_ascend
from tile_kernels.moe.moe_topk_gate_backward_cuda import get_moe_topk_gate_backward_kernel_cuda
from tile_kernels.utils import get_device_guard


def moe_topk_gate_backward(
    scores: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    grad_topk_weights: torch.Tensor,
    routed_scaling_factor: float,
    scoring_func: str = 'sqrtsoftplus',
    mask: Optional[torch.Tensor] = None,
    grad_scores_sum: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the gradient of `moe_topk_gate_forward` w.r.t. `logits`, fused with the optional sequence-level aux-loss term.

    With `u = scores = sqrt(softplus(logits))`, `u_j = u[topk_idx[j]]`, `D = sum_j u_j + 1e-20` and
    `w_j = routed_scaling_factor * u_j / D`, the routed weights give `dL/du_j = (routed_scaling_factor * g_j - <g, w>) / D`.
    The aux loss consumes the per-sequence sums of `u / sum(u)`, whose upstream gradient `h` yields
    `dL/du = (h - <h, u> / sum(u)) / sum(u)` for tokens inside `mask`. Both terms are accumulated in the dense
    expert dimension and multiplied by the sqrt(softplus) derivative recovered from `u`.
    Args:
        scores: The `scores` output of `moe_topk_gate_forward` of shape (num_tokens, num_routed_experts)
        topk_idx: Logical (unmapped) routed expert indices of shape (num_tokens, num_topk); -1 marks masked positions
        topk_weights: Routed expert weights of shape (num_tokens, num_topk)
        grad_topk_weights: Gradient of `topk_weights` of shape (num_tokens, num_topk)
        routed_scaling_factor: Scaling factor for routed expert weights
        scoring_func: Must be 'sqrtsoftplus'; kept in the signature for the callers' config plumbing
        mask: Optional mask to skip specific tokens (True = valid, False = skip)
        grad_scores_sum: Optional gradient of the per-sequence normalized score sums of shape (num_seqs, num_routed_experts);
            `num_tokens` must be divisible by `num_seqs`. Must be provided together with `mask`
        out: Optional pre-allocated grad_logits tensor

    Returns:
        grad_logits: Gradient of `logits` of shape (num_tokens, num_routed_experts)

    Only the routed columns with logical expert indices are supported (no shared-as-routed, physical mapping or force_random).
    """
    assert scoring_func == 'sqrtsoftplus', f'moe_topk_gate_backward only supports sqrtsoftplus scoring, got {scoring_func!r}'
    assert scores.dim() == 2 and scores.is_contiguous() and scores.dtype == torch.float32
    num_tokens = scores.size(0)
    num_routed_experts = scores.size(1)

    assert topk_idx.dim() == 2 and topk_idx.is_contiguous() and topk_idx.dtype == torch.int64
    assert topk_idx.size(0) == num_tokens
    num_topk = topk_idx.size(1)
    assert num_topk <= num_routed_experts, f'num_topk ({num_topk}) must be less than or equal to num_routed_experts ({num_routed_experts}).'
    for t in (topk_weights, grad_topk_weights):
        assert t.size() == topk_idx.size() and t.is_contiguous() and t.dtype == torch.float32

    aux_exists = grad_scores_sum is not None
    assert aux_exists == (mask is not None), 'grad_scores_sum and mask must be both provided or both None'
    if aux_exists:
        assert grad_scores_sum.dim() == 2 and grad_scores_sum.is_contiguous() and grad_scores_sum.dtype == torch.float32
        assert grad_scores_sum.size(1) == num_routed_experts and num_tokens % grad_scores_sum.size(0) == 0
        assert mask.dim() == 1 and mask.is_contiguous() and mask.dtype == torch.bool
        assert mask.size(0) == num_tokens

    if out is None:
        grad_logits = torch.empty_like(scores)
    else:
        grad_logits = out
        assert grad_logits.size() == scores.size() and grad_logits.is_contiguous() and grad_logits.dtype == torch.float32

    if num_tokens == 0:
        return grad_logits

    # The input tensor decides which GPU this runs on; the current device must not.
    with get_device_guard(scores.device):
        if is_ascend():
            from tile_kernels.moe.moe_topk_gate_backward_asc import get_moe_topk_gate_backward_kernel_asc

            kernel = get_moe_topk_gate_backward_kernel_asc(
                num_topk,
                num_routed_experts,
                aux_exists,
            )
        else:
            kernel = get_moe_topk_gate_backward_kernel_cuda(
                num_topk,
                num_routed_experts,
                aux_exists,
                use_pdl=get_pdl(),
            )

        kernel(
            scores, topk_idx, topk_weights, grad_topk_weights,
            mask, grad_scores_sum,
            grad_logits, routed_scaling_factor,
        )  # fmt: off

    return grad_logits

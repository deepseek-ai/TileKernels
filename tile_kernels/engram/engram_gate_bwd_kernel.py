import torch
from tilelang import language as T

from tile_kernels.config import (
    get_num_sms,
    get_num_vec_cores,
    get_pdl,
    is_ascend,
)
from tile_kernels.engram.engram_gate_bwd_cuda import get_engram_gate_bwd_kernel_cuda


def engram_gate_bwd(
    grad_out: torch.Tensor,
    hidden_states: torch.Tensor,
    kv: torch.Tensor,
    weight_xk_fused: torch.Tensor,
    weight_value: torch.Tensor | None,
    dot: torch.Tensor,
    gate_score: torch.Tensor,
    rstd_x: torch.Tensor,
    rstd_kv: torch.Tensor,
    clamp_value: float,
    image_token_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Engram gate backward: computes grad_hidden_states, grad_kv, grad_w_partial.

    Args:
        grad_out: Gradient of output, shape (num_tokens, hc_mult, hidden_size), same dtype as hidden_states.
        hidden_states: Original input from forward, shape (num_tokens, hc_mult, hidden_size), bfloat16 or float32.
        kv: Key/value embeddings from forward, shape (num_tokens, hc_mult + 1, hidden_size), bfloat16.
        weight_xk_fused: Fused RMSNorm weight (weight_hidden * weight_embed), shape (hc_mult, hidden_size), float32.
        weight_value: RMSNorm weight for v, shape (hidden_size,), bfloat16 or float32. Pass None to disable value normalization.
        dot: Saved raw dot product from forward, shape (num_tokens, hc_mult), float32.
        gate_score: Saved gate scores from forward, shape (num_tokens, hc_mult), float32.
        rstd_x: Saved reciprocal std of x from forward, shape (num_tokens, hc_mult), float32.
        rstd_kv: Saved reciprocal std of k and optionally v, shape (num_tokens, hc_mult + has_value_norm), float32.
        clamp_value: Clamp threshold for signed-sqrt gate activation.
        image_token_mask: Optional, ``True`` stands for image tokens, shape (num_tokens,), bool.

    Returns:
        tuple: (grad_hidden_states, grad_kv, grad_w_partial) where
        grad_kv is bfloat16. grad_w_partial needs further reduction over its first dimension.
    """
    _, hc_mult, hidden_size = hidden_states.shape
    has_value_norm = weight_value is not None
    scalar = hidden_size**-0.5
    if is_ascend():
        from tile_kernels.engram.engram_gate_bwd_asc import get_engram_gate_bwd_kernel_asc

        num_partials = get_num_vec_cores()
        kernel = get_engram_gate_bwd_kernel_asc(
            hidden_size,
            scalar,
            clamp_value,
            hc_mult,
            has_image_token_mask=image_token_mask is not None,
            has_value_norm=has_value_norm,
            x_dtype=T.dtype(hidden_states.dtype),
            wv_dtype=T.dtype(weight_value.dtype) if has_value_norm else T.bfloat16,
            num_vec_cores=num_partials,
        )
    else:
        num_partials = get_num_sms()
        kernel = get_engram_gate_bwd_kernel_cuda(
            hidden_size,
            scalar,
            num_partials,
            clamp_value,
            hc_mult,
            has_image_token_mask=image_token_mask is not None,
            has_value_norm=has_value_norm,
            x_dtype=T.dtype(hidden_states.dtype),
            wv_dtype=T.dtype(weight_value.dtype) if has_value_norm else T.bfloat16,
            use_pdl=get_pdl(),
        )

    grad_hidden_states = torch.empty_like(hidden_states)
    grad_kv = torch.empty_like(kv)

    grad_w_partial = torch.empty(
        (num_partials, hc_mult + has_value_norm, hidden_size),
        dtype=torch.float32,
        device=hidden_states.device,
    )

    kernel(
        grad_out,
        hidden_states,
        kv,
        weight_xk_fused,
        weight_value,
        dot,
        gate_score,
        rstd_x,
        rstd_kv,
        image_token_mask,
        grad_hidden_states,
        grad_kv,
        grad_w_partial,
    )
    return grad_hidden_states, grad_kv, grad_w_partial

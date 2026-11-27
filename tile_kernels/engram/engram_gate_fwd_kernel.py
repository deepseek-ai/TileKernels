import torch
from tilelang import language as T

from tile_kernels.config import get_num_vec_cores, get_pdl, is_ascend
from tile_kernels.engram.engram_gate_fwd_cuda import get_engram_gate_fwd_kernel_cuda


def engram_gate_fwd(
    hidden_states: torch.Tensor,
    kv: torch.Tensor,
    weight_xk_fused: torch.Tensor,
    weight_value: torch.Tensor | None,
    eps: float,
    clamp_value: float,
    save_for_backward: bool = True,
    image_token_mask: torch.Tensor | None = None,
    inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Engram gate forward pass.

    Args:
        hidden_states: Input of shape (num_tokens, hc_mult, hidden_size), bfloat16 or float32.
        kv: Key/value embeddings of shape (num_tokens, hc_mult + 1, hidden_size), bfloat16.
        weight_xk_fused: Fused RMSNorm weight (weight_hidden * weight_embed), shape (hc_mult, hidden_size), float32.
        weight_value: RMSNorm weight for v, shape (hidden_size,), bfloat16 or float32. Pass None to disable value normalization.
        eps: Epsilon for RMSNorm numerical stability.
        clamp_value: Clamp threshold for signed-sqrt gate activation.
        save_for_backward: If True, saves dot/gate_score/rstd_x/rstd_kv for backward. If False,
            returns None for those intermediates (inference mode).
        image_token_mask: Optional, 'True' stands for image tokens (num_tokens,), bool.

    Returns:
        tuple: (output, dot, gate_score, rstd_x, rstd_kv). When save_for_backward=False,
            dot/gate_score/rstd_x/rstd_kv are None. When inplace=True, output is ``hidden_states``.
    """
    num_tokens, hc_mult, hidden_size = hidden_states.shape
    scalar = hidden_size**-0.5
    if is_ascend():
        from tile_kernels.engram.engram_gate_fwd_asc import get_engram_gate_fwd_kernel_asc

        kernel = get_engram_gate_fwd_kernel_asc(
            hidden_size,
            eps,
            scalar,
            clamp_value,
            hc_mult,
            save_for_backward,
            image_token_mask is not None,
            weight_value is not None,
            T.dtype(hidden_states.dtype),
            T.dtype(weight_value.dtype) if weight_value is not None else T.bfloat16,
            inplace=inplace,
            num_vec_cores=get_num_vec_cores(),
        )
    else:
        kernel = get_engram_gate_fwd_kernel_cuda(
            hidden_size,
            eps,
            scalar,
            clamp_value,
            hc_mult,
            save_for_backward,
            image_token_mask is not None,
            weight_value is not None,
            T.dtype(hidden_states.dtype),
            T.dtype(weight_value.dtype) if weight_value is not None else T.bfloat16,
            inplace,
            use_pdl=get_pdl(),
        )

    output = hidden_states if inplace else torch.empty_like(hidden_states)
    if save_for_backward:
        dot = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=hidden_states.device)
        gate_score = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=hidden_states.device)
        rstd_x = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=hidden_states.device)
        rstd_kv = torch.empty((num_tokens, hc_mult + (weight_value is not None)), dtype=torch.float32, device=hidden_states.device)
    else:
        dot = gate_score = rstd_x = rstd_kv = None

    if num_tokens > 0:
        kernel(
            hidden_states,
            kv,
            weight_xk_fused,
            weight_value,
            image_token_mask,
            None if inplace else output,
            dot,
            gate_score,
            rstd_x,
            rstd_kv,
        )

    return output, dot, gate_score, rstd_x, rstd_kv

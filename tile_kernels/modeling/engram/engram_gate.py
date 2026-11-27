import torch

from tile_kernels.engram import engram_gate_bwd, engram_gate_fwd, fused_weight, grad_w_reduce


class EngramGateFn(torch.autograd.Function):
    """
    Fused engram gate with RMSNorm.

    Computes:
        gate = sigmoid(signed_sqrt(dot(RMSNorm(x, wh), RMSNorm(k, we)) * scalar))
        output = hidden_states + gate * value

    ``value`` is RMSNorm(v, weight_value) when weight_value is provided, and v otherwise.

    where ``signed_sqrt(x) = sign(x) * sqrt(|x|)`` and ``scalar = 1 / sqrt(hidden_size)``.
    A ``clamp_min`` is also applied.

    Args:
        hidden_states: [*, hc_mult, hidden_size], bf16 or fp32.
        kv:            [*, (hc_mult + 1) * hidden_size], bf16.
        weight_hidden: [hc_mult, hidden_size], bf16 or fp32. RMSNorm weight for x.
        weight_embed:  [hc_mult, hidden_size], same dtype as weight_hidden. RMSNorm weight for k.
        weight_value:  [hidden_size] RMSNorm weight for v, bf16 or fp32. Pass None to disable value normalization.
        clamp_value:   float. Clamp range.
        eps:           float. RMSNorm epsilon.
        image_token_mask: bool tensor over tokens.

    Returns:
        output: same shape and dtype as hidden_states.

    Note:
        If a norm weight has a ``main_grad`` attribute, gradients are accumulated
        into ``main_grad`` in-place and ``None`` is returned for that parameter.
        Otherwise, gradients are returned.
    """

    @staticmethod
    def forward(ctx, hidden_states, kv, weight_hidden, weight_embed, weight_value, clamp_value, eps, image_token_mask):
        origin_shape = hidden_states.shape
        *_, hc_mult, hidden_size = origin_shape

        x = hidden_states.view(-1, hc_mult, hidden_size)
        kv = kv.view(-1, hc_mult + 1, hidden_size)
        if image_token_mask is not None:
            image_token_mask = image_token_mask.view(-1)

        weight_fused = fused_weight(weight_hidden, weight_embed)
        output, dot, gate_score, rstd_x, rstd_kv = engram_gate_fwd(
            x,
            kv,
            weight_fused,
            weight_value,
            eps,
            clamp_value,
            image_token_mask=image_token_mask,
        )

        ctx.save_for_backward(
            x, kv, weight_hidden, weight_embed, weight_value, weight_fused,
            dot, gate_score, rstd_x, rstd_kv, image_token_mask,
        )  # fmt: off
        ctx.clamp_value = clamp_value
        ctx.origin_shape = origin_shape
        return output.view(origin_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (x, kv, weight_hidden, weight_embed, weight_value, weight_fused, dot, gate_score, rstd_x, rstd_kv, image_token_mask) = ctx.saved_tensors
        origin_shape = ctx.origin_shape
        clamp_value = ctx.clamp_value
        *_, hc_mult, hidden_size = origin_shape

        grad_out = grad_output.view(-1, hc_mult, hidden_size)

        grad_x, grad_kv, grad_w_partial = engram_gate_bwd(
            grad_out, x, kv, weight_fused, weight_value,
            dot, gate_score, rstd_x, rstd_kv, clamp_value,
            image_token_mask=image_token_mask,
        )  # fmt: off

        # Use main_grad (fp32 gradient buffer) if available, otherwise allocate fp32 grad tensors.
        # grad_w_reduce accumulates into grad_wh / grad_we in-place.
        main_grad_wh = getattr(weight_hidden, 'main_grad', None)
        main_grad_we = getattr(weight_embed, 'main_grad', None)
        grad_wh = main_grad_wh if main_grad_wh is not None else torch.zeros_like(weight_hidden, dtype=torch.float32)
        grad_we = main_grad_we if main_grad_we is not None else torch.zeros_like(weight_embed, dtype=torch.float32)
        main_grad_wv = getattr(weight_value, 'main_grad', None)
        grad_wv = None
        if weight_value is not None:
            grad_wv = main_grad_wv if main_grad_wv is not None else torch.zeros_like(weight_value, dtype=torch.float32)
        grad_w_reduce(
            grad_w_partial,
            weight_hidden,
            weight_embed,
            grad_wh,
            grad_we,
            grad_wv,
        )

        kv_origin_shape = origin_shape[:-2] + ((hc_mult + 1) * hidden_size,)
        # Return None for weight grads when main_grad is used (already accumulated in-place).
        return (
            grad_x.view(origin_shape),
            grad_kv.view(kv_origin_shape),
            None if main_grad_wh is not None else grad_wh,
            None if main_grad_we is not None else grad_we,
            None if main_grad_wv is not None else grad_wv,
            None,
            None,
            None,
        )


engram_gate = EngramGateFn.apply

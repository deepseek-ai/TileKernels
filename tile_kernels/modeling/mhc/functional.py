import torch
import torch.nn.functional as F

from tile_kernels.mhc.norm_fn_kernel import _mhc_fn_normw_merge_fwd

from .ops.expand import expand_to_mhc
from .ops.head_compute_mix import mhc_head_compute_mix
from .ops.norm_fn import mhc_pre_norm_fn
from .ops.post import mhc_post
from .ops.pre_apply_mix import mhc_pre_apply_mix
from .ops.pre_big_fuse import mhc_pre_big_fuse
from .ops.pre_split_mixes import mhc_pre_split_mixes
from .ops.sinkhorn import sinkhorn_normalize


def expand_from_embedding(x: torch.Tensor, mhc_mult: int = 4) -> torch.Tensor:
    """Expand embedding from (..., H) to (..., mhc_mult, H).

    This is the entry point that converts a standard transformer embedding
    into the multi-head residual format required by MHC.

    Args:
        x: input tensor of shape (..., hidden_size)
        mhc_mult: number of hyper-connection heads (currently only 4 is guaranteed to work)

    Returns:
        residual tensor of shape (..., mhc_mult, hidden_size)
    """
    return expand_to_mhc(x, mhc_mult)


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    mhc_mult: int = 4,
    post_mult_value: float = 1.0,
    pre_eps: float = 1e-6,
    sinkhorn_eps: float = 1e-6,
    sinkhorn_repeat: int = 10,
    n_splits: int = 16,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """MHC pre-processing for one sublayer (attention or FFN).

    Combines pre_norm_fn + pre_split_mixes + sinkhorn_normalize + pre_apply_mix
    into a single call. Automatically uses the fused big_fuse kernel when
    gradients are disabled (inference mode).

    Args:
        residual: MHC residual tensor of shape [..., mhc_mult, hidden_size]
        fn: weight matrix of shape [mhc_mult * (mhc_mult + 2), mhc_mult * hidden_size]
        scale: sigmoid scaling of shape [3]
        base: mix biases of shape [mhc_mult * (mhc_mult + 2)]
        norm_weight: optional RMSNorm weight of shape [mhc_mult * hidden_size]
        norm_eps: epsilon for RMSNorm
        mhc_mult: number of hyper-connection heads (currently only 4 is guaranteed to work)
        post_mult_value: multiplier for post-layer mix
        pre_eps: epsilon for pre-layer mix sigmoid
        sinkhorn_eps: epsilon for Sinkhorn normalization
        sinkhorn_repeat: number of Sinkhorn iterations
        n_splits: number of splits for split-K GEMM

    Returns:
        layer_input: tensor of shape [..., hidden_size], input to the sublayer
        ctx: opaque tuple (post_mix, comb_mix) to pass to mhc_post
    """
    if not torch.is_grad_enabled():
        if norm_weight is not None:
            merged_fn = torch.empty_like(fn)
            _mhc_fn_normw_merge_fwd(*fn.shape)(fn, norm_weight, merged_fn)
            fn = merged_fn
        post_mix, comb_mix, layer_input = mhc_pre_big_fuse(
            residual,
            fn,
            scale,
            base,
            rms_eps=norm_eps,
            mhc_pre_eps=pre_eps,
            mhc_sinkhorn_eps=sinkhorn_eps,
            mhc_post_mult_value=post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
            n_splits=n_splits,
        )
        return layer_input, (post_mix, comb_mix)

    mixes = mhc_pre_norm_fn(
        residual,
        fn,
        norm_weight,
        norm_eps,
        n_splits=n_splits,
    )

    pre_mix, post_mix, comb_mix = mhc_pre_split_mixes(
        mixes,
        scale,
        base,
        mhc_mult,
        post_mult_value,
        pre_eps,
    )

    comb_mix = sinkhorn_normalize(comb_mix, repeat=sinkhorn_repeat, eps=sinkhorn_eps)

    layer_input = mhc_pre_apply_mix(residual, pre_mix)

    return layer_input, (post_mix, comb_mix)


def mhc_head(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    mhc_mult: int = 4,
    pre_eps: float = 1e-6,
    n_splits: int = 16,
) -> torch.Tensor:
    """MHC head processing for the final language model head.

    Combines pre_norm_fn + head_compute_mix + pre_apply_mix into a single call.
    The fn parameter follows the same convention as block-level fn:
    [mhc_mult, mhc_mult * hidden_size], which is padded to [mhc_mult * (mhc_mult + 2), ...]
    internally to reuse the pre_norm_fn kernel.

    Args:
        residual: MHC residual tensor of shape [..., mhc_mult, hidden_size]
        fn: weight matrix of shape [mhc_mult, mhc_mult * hidden_size]
        scale: sigmoid scaling of shape [1] or scalar
        base: mix biases of shape [mhc_mult]
        norm_weight: optional RMSNorm weight of shape [mhc_mult * hidden_size]
        norm_eps: epsilon for RMSNorm
        mhc_mult: number of hyper-connection heads (currently only 4 is guaranteed to work)
        pre_eps: epsilon for pre-layer mix sigmoid
        n_splits: number of splits for split-K GEMM

    Returns:
        layer_input: tensor of shape [..., hidden_size], input to lm_head
    """
    mhc_mult3 = mhc_mult * (2 + mhc_mult)

    if fn.shape[0] < mhc_mult3:
        fn = F.pad(fn, (0, 0, 0, mhc_mult3 - fn.shape[0]))

    mixes = mhc_pre_norm_fn(
        residual,
        fn,
        norm_weight,
        norm_eps,
        n_splits=n_splits,
    )

    mixes = mixes[..., :mhc_mult]

    if scale.numel() == 1:
        scale = scale.reshape(1)
    mix = mhc_head_compute_mix(mixes, scale, base, pre_eps)

    return mhc_pre_apply_mix(residual, mix.unsqueeze(-1))

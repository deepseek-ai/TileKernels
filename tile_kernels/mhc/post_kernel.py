import math

import torch
from tilelang import language as T

from tile_kernels.config import get_max_ub_per_vector_core, get_num_vec_cores, get_pdl, is_ascend
from tile_kernels.mhc.post_cuda import mhc_post_bwd_cuda, mhc_post_fwd_cuda
from tile_kernels.utils import align, ceil_div

# Safety margin below the 248KB AIV UB limit; the bf16 kernel already runs at
# ~229KB for hidden=4096, so stay at or below that footprint.
_ASC_UB_RESERVE_BYTES = 8 * 1024
_ASC_POST_FWD_NUM_STAGES = 3


def _asc_post_mix_bytes(mhc: int, num_stages: int) -> int:
    """UB bytes of the fp32 mix buffers that are versioned with ``num_stages``."""
    return (mhc * mhc + align(mhc, 8)) * 4 * num_stages


def _asc_post_bwd_acc_bytes(mhc: int) -> int:
    """UB bytes of the single-version fp32 gradient accumulators of the bwd kernel."""
    from tile_kernels.mhc.common_asc import _VEC

    return (mhc + mhc * mhc) * _VEC * 4 + (align(mhc, 8) + mhc * mhc) * 4


def _asc_post_ub_bytes(h_blk: int, per_h_blk_bytes: int, num_stages: int, fixed_bytes: int) -> int:
    return per_h_blk_bytes * h_blk * num_stages + fixed_bytes


def _asc_post_h_blk(hidden: int, per_h_blk_bytes: int, num_stages: int, fixed_bytes: int) -> int:
    """Pick the Ascend hidden block for the mhc post kernels.

    Keep the number of hidden tiles minimal and split ``hidden`` evenly. Prefer a
    block that divides ``hidden``: the backward kernel folds every lane of a tile
    into the fp32 mix accumulators, so a clamped tail tile would add stale UB
    lanes to ``d_post_layer_mix`` / ``d_comb_res_mix``.
    """
    from tile_kernels.mhc.common_asc import _H_BLK_ALIGN

    available = get_max_ub_per_vector_core(use_simt=False) - _ASC_UB_RESERVE_BYTES - fixed_bytes
    max_h_blk = max(available // (per_h_blk_bytes * num_stages) // _H_BLK_ALIGN * _H_BLK_ALIGN, _H_BLK_ALIGN)
    h_blk = align(ceil_div(hidden, ceil_div(hidden, max_h_blk)), _H_BLK_ALIGN)
    if hidden % h_blk == 0:
        return h_blk
    for candidate in range(h_blk - _H_BLK_ALIGN, _H_BLK_ALIGN - 1, -_H_BLK_ALIGN):
        if hidden % candidate == 0:
            return candidate
    return h_blk


def _asc_post_fwd_h_blk(hidden: int, mhc: int, elem_bytes: int) -> int:
    # residual (mhc rows) + x (1 row) + out (mhc rows), versioned num_stages.
    fixed_bytes = _asc_post_mix_bytes(mhc, _ASC_POST_FWD_NUM_STAGES)
    return _asc_post_h_blk(hidden, (2 * mhc + 1) * elem_bytes, _ASC_POST_FWD_NUM_STAGES, fixed_bytes)


def _asc_post_bwd_h_blk_and_num_stages(hidden: int, mhc: int, elem_bytes: int) -> tuple[int, int]:
    # d_o (mhc rows) + x (1) + residual (mhc) + d_x (1) + d_residual (mhc).
    per_h_blk_bytes = (3 * mhc + 2) * elem_bytes
    acc_bytes = _asc_post_bwd_acc_bytes(mhc)
    # Prefer the wider blocks of 2 stages, then take the third stage when it fits.
    num_stages = 2
    h_blk = _asc_post_h_blk(hidden, per_h_blk_bytes, num_stages, _asc_post_mix_bytes(mhc, num_stages) + acc_bytes)
    limit = get_max_ub_per_vector_core(use_simt=False) - _ASC_UB_RESERVE_BYTES
    if _asc_post_ub_bytes(h_blk, per_h_blk_bytes, 3, _asc_post_mix_bytes(mhc, 3) + acc_bytes) <= limit:
        num_stages = 3
    return h_blk, num_stages


def mhc_post_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    num_seqs, num_tokens, mhc, hidden = residual.shape

    assert post_layer_mix.dtype == torch.float32, f'{post_layer_mix.dtype=}'
    assert comb_res_mix.dtype == torch.float32, f'{comb_res_mix.dtype=}'
    assert x.shape == (num_seqs, num_tokens, hidden), f'{x.shape=}'
    assert post_layer_mix.shape == (num_seqs, num_tokens, mhc, 1), f'{post_layer_mix.shape=}'
    assert comb_res_mix.shape == (num_seqs, num_tokens, mhc, mhc), f'{comb_res_mix.shape=}'

    residual = residual.contiguous()
    assert x.is_contiguous()
    assert post_layer_mix.is_contiguous()
    assert comb_res_mix.is_contiguous()

    if out is None:
        out = torch.empty_like(residual)

    n = num_seqs * num_tokens
    if is_ascend():
        from tile_kernels.mhc.post_asc import mhc_post_fwd_asc

        h_blk = _asc_post_fwd_h_blk(hidden, mhc, x.dtype.itemsize)
        mhc_post_fwd_asc(
            comb_res_mix.view(n, mhc, mhc),
            residual.view(n, mhc, hidden),
            post_layer_mix.view(n, mhc),
            x.view(n, hidden),
            out.view(n, mhc, hidden),
            h_blk=h_blk,
            num_vec_cores=get_num_vec_cores(),
            x_dtype=T.dtype(x.dtype),
        )
    else:
        h_blk = math.gcd(hidden, 1024)
        vec = math.gcd(8, h_blk // 128)
        assert h_blk % 128 == 0

        mhc_post_fwd_cuda(
            comb_res_mix.flatten(0, 1),
            residual.flatten(0, 1),
            post_layer_mix.flatten(0, 1).squeeze(-1),
            x.flatten(0, 1),
            out.flatten(0, 1),
            vec=vec,
            n_thr=128,
            h_blk=h_blk,
            x_dtype=T.dtype(x.dtype),
            use_pdl=get_pdl(),
        )
    return out


def mhc_post_bwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    d_o: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = d_o.shape[0] * d_o.shape[1]
    mhc = d_o.shape[2]
    h = d_o.shape[3]

    if is_ascend():
        from tile_kernels.mhc.post_asc import mhc_post_bwd_asc

        h_blk, num_stages = _asc_post_bwd_h_blk_and_num_stages(h, mhc, x.dtype.itemsize)
        d_comb_res_mix, d_residual, d_post_layer_mix, d_x = mhc_post_bwd_asc(
            d_o.contiguous().view(n, mhc, h),
            comb_res_mix.view(n, mhc, mhc),
            residual.view(n, mhc, h),
            post_layer_mix.view(n, mhc),
            x.view(n, h),
            h_blk=h_blk,
            num_stages=num_stages,
            num_vec_cores=get_num_vec_cores(),
            x_dtype=T.dtype(x.dtype),
        )
    else:
        h_blk = math.gcd(h, 512)
        d_comb_res_mix, d_residual, d_post_layer_mix, d_x = mhc_post_bwd_cuda(
            d_o.contiguous().view(n, mhc, h),
            comb_res_mix.view(n, mhc, mhc),
            residual.view(n, mhc, h),
            post_layer_mix.view(n, mhc),
            x.view(n, h),
            n_thr=64,
            h_blk=h_blk,
            x_dtype=T.dtype(x.dtype),
            use_pdl=get_pdl(),
        )
    assert isinstance(d_x, torch.Tensor)
    assert isinstance(d_post_layer_mix, torch.Tensor)
    assert isinstance(d_comb_res_mix, torch.Tensor)
    assert isinstance(d_residual, torch.Tensor)

    return (
        d_x.view_as(x),
        d_residual.view_as(residual),
        d_post_layer_mix.view_as(post_layer_mix),
        d_comb_res_mix.view_as(comb_res_mix),
    )

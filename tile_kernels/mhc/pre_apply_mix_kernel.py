import math

from tilelang import language as T

from tile_kernels.config import get_max_ub_per_vector_core, get_num_vec_cores, get_pdl, is_ascend
from tile_kernels.mhc.pre_apply_mix_cuda import mhc_pre_apply_mix_bwd_cuda, mhc_pre_apply_mix_fwd_cuda
from tile_kernels.utils import align, ceil_div


def _asc_pick_h_blk(hidden: int, max_h_blk: int) -> int:
    """Largest ``_H_BLK_ALIGN``-aligned tile that fits ``max_h_blk``, preferring an exact divisor of ``hidden``."""
    from tile_kernels.mhc.common_asc import _H_BLK_ALIGN

    max_h_blk = max_h_blk // _H_BLK_ALIGN * _H_BLK_ALIGN
    if max_h_blk < _H_BLK_ALIGN:
        raise ValueError(f'pre_apply_mix: UB budget only allows h_blk={max_h_blk} < {_H_BLK_ALIGN}')
    if hidden <= max_h_blk:
        return align(hidden, _H_BLK_ALIGN)
    h_blk = max_h_blk
    while h_blk > _H_BLK_ALIGN and hidden % h_blk:
        h_blk -= _H_BLK_ALIGN
    return h_blk


def _asc_pre_apply_mix_fwd_h_blk(hidden: int, mhc: int, x_dtype: T.dtype) -> int:
    """Hidden tile for `mhc_pre_apply_mix_fwd_asc` (its `num_stages` is fixed at 3)."""
    from tile_kernels.mhc.common_asc import _H_BLK_ALIGN

    if x_dtype == T.bfloat16:
        # Historical bf16 tiling, left untouched so bf16 behaviour and perf do not move.
        h_blk = hidden if hidden <= 6144 else min(ceil_div(hidden, 2), 6144)
        return align(h_blk, _H_BLK_ALIGN)

    # Every stage holds x_ub (mhc, h_blk) plus o_ub (h_blk,); the mixes are fp32 and tiny.
    num_stages = 3
    mix_bytes = ceil_div(mhc, 8) * 8 * T.float32.bytes * num_stages
    ub_budget = get_max_ub_per_vector_core(use_simt=False)
    max_h_blk = (ub_budget - mix_bytes) // ((mhc + 1) * x_dtype.bytes * num_stages)
    h_blk = _asc_pick_h_blk(hidden, max_h_blk)
    assert (mhc + 1) * h_blk * x_dtype.bytes * num_stages + mix_bytes <= ub_budget
    return h_blk


def _asc_pre_apply_mix_bwd_tiling(hidden: int, mhc: int, x_dtype: T.dtype) -> tuple[int, int]:
    """``(h_blk, num_stages)`` for `mhc_pre_apply_mix_bwd_asc`."""
    from tile_kernels.mhc.common_asc import _H_BLK_ALIGN, _VEC

    if x_dtype == T.bfloat16:
        # Historical bf16 tiling, left untouched so bf16 behaviour and perf do not move.
        if hidden <= 4096:
            return hidden, 3
        return align(min(ceil_div(hidden, 4), 3072), _H_BLK_ALIGN), 4

    # Each stage keeps o_grad_ub (h_blk,) in addition to x_ub and x_grad_ub (mhc, h_blk).
    num_stages = 3 if hidden <= 4096 else 4
    mix_bytes = 2 * ceil_div(mhc, 8) * 8 * T.float32.bytes * num_stages + mhc * _VEC * T.float32.bytes
    ub_budget = get_max_ub_per_vector_core(use_simt=False)
    max_h_blk = (ub_budget - mix_bytes) // ((2 * mhc + 1) * x_dtype.bytes * num_stages)
    h_blk = _asc_pick_h_blk(hidden, max_h_blk)
    assert (2 * mhc + 1) * h_blk * x_dtype.bytes * num_stages + mix_bytes <= ub_budget
    return h_blk, num_stages


def mhc_pre_apply_mix_fwd(x, mix, o):
    if is_ascend():
        from tile_kernels.mhc.pre_apply_mix_asc import mhc_pre_apply_mix_fwd_asc

        num_tokens, mhc, hidden = x.shape
        x_dtype = T.dtype(x.dtype)
        mhc_pre_apply_mix_fwd_asc(
            x,
            mix,
            o,
            h_blk=_asc_pre_apply_mix_fwd_h_blk(hidden, mhc, x_dtype),
            num_vec_cores=get_num_vec_cores(),
            x_dtype=x_dtype,
        )
    else:
        hidden = x.shape[-1]
        mhc_pre_apply_mix_fwd_cuda(
            x,
            mix,
            o,
            n_thr=64,
            h_blk=math.gcd(1024, hidden),
            x_dtype=T.dtype(x.dtype),
            use_pdl=get_pdl(),
        )


def mhc_pre_apply_mix_bwd(o_grad, x, mix, x_grad):
    if is_ascend():
        from tile_kernels.mhc.pre_apply_mix_asc import mhc_pre_apply_mix_bwd_asc

        hidden, mhc = x.shape[2], x.shape[1]
        x_dtype = T.dtype(x.dtype)
        h_blk, num_stages = _asc_pre_apply_mix_bwd_tiling(hidden, mhc, x_dtype)
        return mhc_pre_apply_mix_bwd_asc(
            o_grad,
            x,
            mix,
            x_grad,
            h_blk=h_blk,
            num_stages=num_stages,
            num_vec_cores=get_num_vec_cores(),
            x_dtype=x_dtype,
        )
    else:
        hidden = x.shape[-1]
        return mhc_pre_apply_mix_bwd_cuda(
            o_grad,
            x,
            mix,
            x_grad,
            n_thr=128,
            h_blk=math.gcd(1024, hidden),
            x_dtype=T.dtype(x.dtype),
            use_pdl=get_pdl(),
        )

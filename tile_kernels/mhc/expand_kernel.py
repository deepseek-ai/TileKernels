import torch
from tilelang import language as T

from tile_kernels.config import get_max_ub_per_vector_core, get_num_vec_cores, get_pdl, is_ascend
from tile_kernels.mhc.expand_cuda import expand_to_mhc_bwd_cuda, expand_to_mhc_fwd_cuda

_ASC_FWD_NUM_STAGES = 7
_ASC_BWD_NUM_STAGES = 3
# Leave a little UB for pipeline flags / allocation granularity on top of the activation tiles.
_ASC_UB_SLACK_BYTES = 16 * 1024


def _asc_fwd_h_blk(hidden: int, mhc_mult: int, x_dtype: torch.dtype) -> int:
    # Activation footprint per stage is (x tile + mhc_mult o tiles), doubled/quadrupled by dtype.
    ub_budget_bytes = get_max_ub_per_vector_core(use_simt=False)
    max_h_blk = ub_budget_bytes // ((mhc_mult + 1) * T.dtype(x_dtype).bytes * _ASC_FWD_NUM_STAGES)
    h_blk = min(hidden, max_h_blk // 128 * 128)
    while h_blk > 128 and hidden % h_blk != 0:
        h_blk -= 128
    return h_blk


def _asc_h_blk(hidden: int, mhc_mult: int, x_dtype: torch.dtype) -> int:
    from tile_kernels.mhc.common_asc import _H_BLK_ALIGN

    # Keep h_blk small enough that (mhc_mult + 1) activation tiles fit in UB with multi-stage.
    h_blk = hidden if hidden <= 4096 else (hidden + 1) // 2
    h_blk = (h_blk + _H_BLK_ALIGN - 1) // _H_BLK_ALIGN * _H_BLK_ALIGN
    # Cap at 4096 to fit UB budget for mhc * h_blk * num_stages
    while h_blk > 4096 and hidden % (h_blk // 2) == 0:
        h_blk //= 2
    # fp32 doubles the per-element footprint: the legacy 4096 tile would take 240KB at
    # h=4096/8192, so shrink to the largest _H_BLK_ALIGN multiple that still fits the byte budget.
    ub_budget_bytes = get_max_ub_per_vector_core(use_simt=False) - _ASC_UB_SLACK_BYTES
    max_h_blk = ub_budget_bytes // ((mhc_mult + 1) * T.dtype(x_dtype).bytes * _ASC_BWD_NUM_STAGES)
    if h_blk > max_h_blk:
        h_blk = max_h_blk // _H_BLK_ALIGN * _H_BLK_ALIGN
        while h_blk > _H_BLK_ALIGN and hidden % h_blk != 0:
            h_blk -= _H_BLK_ALIGN
    return h_blk


def expand_to_mhc_fwd(x, o):
    if is_ascend():
        from tile_kernels.mhc.expand_asc import expand_to_mhc_fwd_asc

        hidden = x.shape[-1]
        assert hidden % 128 == 0, f'Ascend expand fwd requires hidden divisible by 128, got {hidden}'
        h_blk = _asc_fwd_h_blk(hidden, o.shape[-2], x.dtype)
        assert hidden % h_blk == 0, f'Ascend expand fwd requires hidden divisible by h_blk, got {hidden=}, {h_blk=}'
        expand_to_mhc_fwd_asc(
            x,
            o,
            h_blk=h_blk,
            num_stages=_ASC_FWD_NUM_STAGES,
            num_vec_cores=get_num_vec_cores(),
            x_dtype=T.dtype(x.dtype),
        )
    else:
        expand_to_mhc_fwd_cuda(x, o, dtype=T.dtype(x.dtype), use_pdl=get_pdl())


def expand_to_mhc_bwd(o_grad, x_grad):
    if is_ascend():
        from tile_kernels.mhc.common_asc import _H_BLK_ALIGN
        from tile_kernels.mhc.expand_asc import expand_to_mhc_bwd_asc

        hidden = x_grad.shape[-1]
        assert hidden % _H_BLK_ALIGN == 0, f'Ascend expand bwd requires hidden divisible by {_H_BLK_ALIGN}, got {hidden}'
        h_blk = _asc_h_blk(hidden, o_grad.shape[-2], o_grad.dtype)
        assert hidden % h_blk == 0, f'Ascend expand bwd requires hidden divisible by h_blk, got {hidden=}, {h_blk=}'
        expand_to_mhc_bwd_asc(
            o_grad,
            x_grad,
            h_blk=h_blk,
            num_stages=_ASC_BWD_NUM_STAGES,
            num_vec_cores=get_num_vec_cores(),
            x_dtype=T.dtype(o_grad.dtype),
        )
    else:
        expand_to_mhc_bwd_cuda(o_grad, x_grad, dtype=T.dtype(o_grad.dtype), use_pdl=get_pdl())

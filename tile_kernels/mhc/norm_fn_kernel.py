from tile_kernels.config import get_num_vec_cores, get_pdl, is_ascend
from tile_kernels.mhc.norm_fn_cuda import (
    mhc_fn_normw_merge_bwd_cuda,
)
from tile_kernels.mhc.norm_fn_cuda import (
    mhc_fn_normw_merge_fwd_cuda,
)
from tile_kernels.mhc.norm_fn_cuda import (
    mhc_reduce_partials_and_rmsnorm_bwd_cuda,
)
from tile_kernels.mhc.norm_fn_cuda import (
    mhc_reduce_partials_and_rmsnorm_fwd_cuda,
)

DEFAULT_N_SPLITS = 16


def default_n_splits(k: int) -> int:
    """Default split-K count for the mHC prenorm GEMM."""
    return min(DEFAULT_N_SPLITS, k // 256) if is_ascend() else DEFAULT_N_SPLITS


# ---------------------------------------------------------------------------
# Wrapper functions: single entry point for callers.
# ---------------------------------------------------------------------------


def mhc_fn_normw_merge_fwd(fn, normw, out_fn):
    if is_ascend():
        from tile_kernels.mhc.norm_fn_asc import mhc_fn_normw_merge_fwd_asc

        mhc_fn_normw_merge_fwd_asc(
            fn,
            normw,
            out_fn,
            mhc_mult3=fn.shape[0],
            hidden_block=256,
            num_stages=3,
            num_vec_cores=get_num_vec_cores(),
        )
    else:
        mhc_fn_normw_merge_fwd_cuda(fn, normw, out_fn, use_pdl=get_pdl())


def mhc_fn_normw_merge_bwd(fn, normw, out_fn_grad, fn_grad, normw_grad):
    if is_ascend():
        from tile_kernels.mhc.norm_fn_asc import mhc_fn_normw_merge_bwd_asc

        mhc_fn_normw_merge_bwd_asc(
            fn,
            normw,
            out_fn_grad,
            fn_grad,
            normw_grad,
            mhc_mult3=fn.shape[0],
            hidden_block=128,
            num_stages=3,
            num_vec_cores=get_num_vec_cores(),
        )
    else:
        mhc_fn_normw_merge_bwd_cuda(
            fn,
            normw,
            out_fn_grad,
            fn_grad,
            normw_grad,
            use_pdl=False,
        )  # PDL regression: >=2us slower, keep PDL off.


def mhc_reduce_partials_and_rmsnorm_fwd(
    out_mul_splitted,
    sqrsum_splitted,
    out_mul,
    sqrsum,
    out,
    rms_group_size: int,
    rms_eps: float,
    n_splits: int,
):
    if is_ascend():
        from tile_kernels.mhc.norm_fn_asc import mhc_reduce_partials_and_rmsnorm_fwd_asc

        token_block = 32 if out_mul_splitted.shape[1] >= 2048 else 16
        mhc_reduce_partials_and_rmsnorm_fwd_asc(
            out_mul_splitted,
            sqrsum_splitted,
            out_mul,
            sqrsum,
            out,
            rms_group_size,
            rms_eps,
            n_splits,
            get_num_vec_cores(),
            token_block=token_block,
        )
    else:
        mhc_reduce_partials_and_rmsnorm_fwd_cuda(
            out_mul_splitted,
            sqrsum_splitted,
            out_mul,
            sqrsum,
            out,
            rms_group_size,
            rms_eps,
            n_splits,
            use_pdl=get_pdl(),
        )


def mhc_reduce_partials_and_rmsnorm_bwd(
    out_grad,
    out_mul,
    sqrsum,
    out_mul_grad,
    sqrsum_grad,
    rms_group_size: int,
    rms_eps: float,
):
    if is_ascend():
        from tile_kernels.mhc.norm_fn_asc import mhc_reduce_partials_and_rmsnorm_bwd_asc

        mhc_reduce_partials_and_rmsnorm_bwd_asc(
            out_grad,
            out_mul,
            sqrsum,
            out_mul_grad,
            sqrsum_grad,
            rms_group_size,
            rms_eps,
            get_num_vec_cores(),
        )
    else:
        mhc_reduce_partials_and_rmsnorm_bwd_cuda(
            out_grad,
            out_mul,
            sqrsum,
            out_mul_grad,
            sqrsum_grad,
            rms_group_size,
            rms_eps,
            use_pdl=get_pdl(),
        )

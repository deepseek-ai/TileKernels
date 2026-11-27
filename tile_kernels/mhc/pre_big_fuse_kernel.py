import math

import torch
from tilelang import language as T

from tile_kernels.config import get_num_vec_cores, get_pdl, is_ascend
from tile_kernels.mhc.pre_big_fuse_cuda import mhc_pre_big_fuse_fwd_cuda

# Ascend fused kernel token block; num_tokens must be a multiple of it.
_ASCEND_TOKEN_BLOCK = 16


def mhc_pre_big_fuse_fwd(
    gemm_out_mul,
    gemm_out_sqrsum,
    mhc_scale,
    mhc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
):
    assert residual.dtype in (torch.bfloat16, torch.float32), f'Unsupported residual dtype: {residual.dtype}'
    assert layer_input.dtype == residual.dtype, f'Expected layer_input.dtype == residual.dtype ({residual.dtype}), got {layer_input.dtype}'
    hidden_size = residual.shape[-1]
    x_dtype = T.dtype(residual.dtype)
    if is_ascend():
        mhc_mult = residual.shape[-2]
        assert mhc_mult == 4, f'Ascend pre_big_fuse only supports mHC4, got {mhc_mult}'
        from tile_kernels.mhc.pre_big_fuse_asc import mhc_pre_big_fuse_fwd_asc

        max_h_blk = 128 if residual.dtype == torch.float32 else 256
        h_blk = min(math.gcd(512, hidden_size), max_h_blk)
        mhc_pre_big_fuse_fwd_asc(
            gemm_out_mul,
            gemm_out_sqrsum,
            mhc_scale,
            mhc_base,
            residual,
            post_mix,
            comb_mix,
            layer_input,
            rms_eps,
            mhc_pre_eps,
            mhc_sinkhorn_eps,
            mhc_post_mult_value,
            sinkhorn_repeat,
            h_blk=h_blk,
            n_splits=n_splits,
            token_block=_ASCEND_TOKEN_BLOCK,
            num_stages=4,
            num_vec_cores=get_num_vec_cores(),
            x_dtype=x_dtype,
        )
    else:
        max_h_blk = 256 if residual.dtype == torch.float32 else 512
        hidden_block = min(math.gcd(512, hidden_size), max_h_blk)
        mhc_pre_big_fuse_fwd_cuda(
            gemm_out_mul,
            gemm_out_sqrsum,
            mhc_scale,
            mhc_base,
            residual,
            post_mix,
            comb_mix,
            layer_input,
            rms_eps,
            mhc_pre_eps,
            mhc_sinkhorn_eps,
            mhc_post_mult_value,
            sinkhorn_repeat,
            hidden_block,
            n_splits=n_splits,
            x_dtype=x_dtype,
            use_pdl=get_pdl(),
        )

import tilelang
import torch
from tilelang import language as T

# HIP has no named barriers, and a wavefront is 64 threads: the 96-thread split
# below (32 + 64) cannot fence shared memory inside its second branch.
_IS_HIP = torch.version.hip is not None

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def mhc_pre_big_fuse_fwd_cuda(
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
    hidden_block: int,
    n_splits: int = 16,
    use_pdl: bool = False,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult = T.const('mhc_mult')
    mhc_mult3 = T.const('mhc_mult3')
    hidden_size = T.const('hidden_size')

    gemm_out_mul: T.Tensor[(n_splits, num_tokens, mhc_mult3), T.float32]
    gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), T.float32]
    mhc_scale: T.Tensor[(3,), T.float32]
    mhc_base: T.Tensor[(mhc_mult3,), T.float32]
    residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16]
    post_mix: T.Tensor[(num_tokens, mhc_mult), T.float32]
    comb_mix: T.Tensor[(num_tokens, mhc_mult * mhc_mult), T.float32]
    layer_input: T.Tensor[(num_tokens, hidden_size), T.bfloat16]

    with T.Kernel(num_tokens, threads=96) as pid:
        if use_pdl:
            T.pdl_sync()
        ##################################################################
        # _mhc_reduce_partials_and_rmsnorm_fwd
        mixes_shared = T.alloc_shared(mhc_mult3, T.float32)
        if T.get_thread_binding() < 32:
            rms = T.alloc_fragment(1, T.float32)
            mixes = T.alloc_fragment(mhc_mult3, T.float32)
            T.clear(mixes)
            rms[0] = 0
            for i_split in T.serial(n_splits):
                rms[0] += gemm_out_sqrsum[i_split, pid]
            rms[0] = T.rsqrt(rms[0] / (mhc_mult * hidden_size) + rms_eps)
            for j in T.Parallel(mhc_mult3):
                mixes[j] = 0
                for i_split in T.serial(n_splits):
                    mixes[j] += gemm_out_mul[i_split, pid, j]
                mixes[j] *= rms[0]
            T.copy(mixes, mixes_shared, disable_tma=True)

        if T.get_thread_binding() < 32:
            ##################################################################
            # _mhc_pre_split_mixes_fwd (post & comb)
            cm = T.alloc_fragment((mhc_mult, mhc_mult), T.float32)
            for j in T.Parallel(mhc_mult):
                post_mix[pid, j] = T.sigmoid(mixes_shared[j + mhc_mult] * mhc_scale[1] + mhc_base[j + mhc_mult]) * mhc_post_mult_value
            for j, k in T.Parallel(mhc_mult, mhc_mult):
                cm[j, k] = mixes_shared[j * mhc_mult + k + mhc_mult * 2] * mhc_scale[2] + mhc_base[j * mhc_mult + k + mhc_mult * 2]

            ##################################################################
            # _mhc_sinkhorn_fwd
            row_sum = T.alloc_fragment(mhc_mult, T.float32)
            col_sum = T.alloc_fragment(mhc_mult, T.float32)

            # comb = comb.softmax(-1) + eps
            row_max = T.alloc_fragment(mhc_mult, T.float32)
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(mhc_mult, mhc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(mhc_mult, mhc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + mhc_sinkhorn_eps

            # comb = comb / (comb.sum(-2) + eps)
            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(mhc_mult, mhc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)

            for _ in T.serial(sinkhorn_repeat - 1):
                # comb = comb / (comb.sum(-1) + eps)
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + mhc_sinkhorn_eps)

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)

            # save comb_mix to global memory
            for j, k in T.Parallel(mhc_mult, mhc_mult):
                comb_mix[pid, j * mhc_mult + k] = cm[j, k]
        elif _IS_HIP:
            ##################################################################
            # _mhc_pre_split_mixes_fwd (pre) + _mhc_pre_apply_mix_fwd, register only:
            # every thread recomputes the pre mix and loads its own residual slice.
            pre_local = T.alloc_local(mhc_mult, T.float32)
            for j in T.serial(mhc_mult):
                pre_local[j] = T.sigmoid(mixes_shared[j] * mhc_scale[0] + mhc_base[j]) + mhc_pre_eps
            for i0_h in T.serial(hidden_size // hidden_block):
                xl = T.alloc_fragment((mhc_mult, hidden_block), T.float32)
                T.copy(residual[pid, 0, i0_h * hidden_block], xl, disable_tma=True)

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)

                for i_mhc in T.serial(mhc_mult):
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre_local[i_mhc] * xl[i_mhc, i1_h]

                T.copy(ol, layer_input[pid, i0_h * hidden_block], disable_tma=True)
        else:
            ##################################################################
            # _mhc_pre_split_mixes_fwd (pre)
            pre_mix_shared = T.alloc_shared(mhc_mult, T.float32)
            for j in T.Parallel(mhc_mult):
                pre_mix_shared[j] = (
                    T.sigmoid(
                        mixes_shared[j] * mhc_scale[0] + mhc_base[j],
                    )
                    + mhc_pre_eps
                )
            ###################################################################
            # _mhc_pre_apply_mix_fwd
            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                xs = T.alloc_shared((mhc_mult, hidden_block), T.bfloat16)
                xl = T.alloc_fragment((mhc_mult, hidden_block), T.float32)
                T.copy(residual[pid, 0, i0_h * hidden_block], xs, disable_tma=True)
                T.copy(xs, xl, disable_tma=True)

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)

                for i_mhc in T.serial(mhc_mult):
                    pre = pre_mix_shared[i_mhc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i_mhc, i1_h]

                T.copy(ol, layer_input[pid, i0_h * hidden_block], disable_tma=True)
        if use_pdl:
            T.pdl_trigger()

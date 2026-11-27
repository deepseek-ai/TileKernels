import tilelang
from tilelang import language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def mhc_pre_apply_mix_fwd_cuda(
    x,
    mix,
    o,
    n_thr: int = 128,
    h_blk: int = 1024,
    x_dtype: T.dtype = T.bfloat16,
    use_pdl: bool = False,
):
    assert x_dtype in (T.bfloat16, T.float32)
    n = T.dynamic('n')
    mhc = T.const('mhc')
    h = T.const('h')

    x: T.Tensor[(n, mhc, h), x_dtype]
    mix: T.Tensor[(n, mhc), T.float32]
    o: T.Tensor[(n, h), x_dtype]

    h_ctas = h // h_blk
    with T.Kernel(n * h_ctas, threads=n_thr) as pid:
        pid_n = pid // h_ctas
        pid_h = pid % h_ctas
        if use_pdl:
            T.pdl_sync()
        mixl = T.alloc_fragment(mhc, T.float32)
        T.copy(mix[pid_n, 0], mixl)

        xs = T.alloc_shared((mhc, h_blk), x_dtype)
        xl = T.alloc_fragment((mhc, h_blk), T.float32)
        T.copy(x[pid_n, 0, pid_h * h_blk], xs, disable_tma=True)
        T.copy(xs, xl, disable_tma=True)

        os = T.alloc_shared(h_blk, x_dtype)
        ol = T.alloc_fragment(h_blk, T.float32)
        T.clear(ol)

        for i_mhc in T.serial(mhc):
            for i1_h in T.Parallel(h_blk):
                ol[i1_h] += mixl[i_mhc] * xl[i_mhc, i1_h]

        T.copy(ol, os, disable_tma=True)
        T.copy(os, o[pid_n, pid_h * h_blk], disable_tma=True)
        if use_pdl:
            T.pdl_trigger()


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def mhc_pre_apply_mix_bwd_cuda(
    o_grad,
    x,
    mix,
    x_grad,
    n_thr: int = 128,
    h_blk: int = 1024,
    x_dtype: T.dtype = T.bfloat16,
    use_pdl: bool = False,
):
    assert x_dtype in (T.bfloat16, T.float32)
    n = T.dynamic('n')
    mhc = T.const('mhc')
    h = T.const('h')

    o_grad: T.Tensor[(n, h), x_dtype]
    x: T.Tensor[(n, mhc, h), x_dtype]
    mix: T.Tensor[(n, mhc), T.float32]
    x_grad: T.Tensor[(n, mhc, h), x_dtype]

    mix_grad = T.empty((n, mhc), T.float32)

    with T.Kernel(n, threads=n_thr) as pid_n:
        if use_pdl:
            T.pdl_sync()
        mixl = T.alloc_fragment(mhc, T.float32)
        T.copy(mix[pid_n, :], mixl, disable_tma=True)

        mgl_reducer = T.alloc_reducer(mhc, T.float32, op='sum')
        mgl_result = T.alloc_fragment(mhc, T.float32)
        T.reducer_init(mgl_reducer)

        for i0_h in T.Pipelined(h // h_blk, num_stages=2):
            ogs = T.alloc_shared(h_blk, x_dtype)
            ogl = T.alloc_fragment(h_blk, T.float32)
            T.copy(o_grad[pid_n, i0_h * h_blk], ogs, disable_tma=True)
            T.copy(ogs, ogl, disable_tma=True)

            xs = T.alloc_shared((mhc, h_blk), x_dtype)
            xl = T.alloc_fragment((mhc, h_blk), T.float32)
            T.copy(x[pid_n, 0, i0_h * h_blk], xs, disable_tma=True)
            T.copy(xs, xl, disable_tma=True)

            xgs = T.alloc_shared((mhc, h_blk), x_dtype)
            xgl = T.alloc_fragment((mhc, h_blk), T.float32)
            T.copy(x_grad[pid_n, 0, i0_h * h_blk], xgs, disable_tma=True)
            T.copy(xgs, xgl, disable_tma=True)

            for i_mhc, i1_h in T.Parallel(mhc, h_blk):
                T.reducer_update(mgl_reducer[i_mhc], ogl[i1_h] * xl[i_mhc, i1_h])
                xgl[i_mhc, i1_h] += mixl[i_mhc] * ogl[i1_h]

            T.copy(xgl, x_grad[pid_n, 0, i0_h * h_blk], disable_tma=True)

        T.finalize_reducer(mgl_reducer, mgl_result)
        T.copy(mgl_result, mix_grad[pid_n, 0], disable_tma=True)
        if use_pdl:
            T.pdl_trigger()

    return mix_grad

import tilelang
from tilelang import language as T


@tilelang.jit
def expand_to_mhc_fwd_cuda(x, o, dtype: T.dtype = T.bfloat16, use_pdl: bool = False):
    assert dtype in (T.bfloat16, T.float32)
    n = T.dynamic('n')
    h, mhc = T.const('h, mhc')
    x: T.Tensor[(n, h), dtype]
    o: T.Tensor[(n, mhc, h), dtype]

    blk_n = 32
    blk_h = 128

    with T.Kernel(T.ceildiv(n, blk_n), T.ceildiv(h, blk_h)) as (pid_i, pid_j):
        if use_pdl:
            T.pdl_sync()
        xl = T.alloc_fragment((blk_n, blk_h), dtype)
        T.copy(x[pid_i * blk_n, pid_j * blk_h], xl)
        for m in T.serial(mhc):
            for ti, tj in T.Parallel(blk_n, blk_h):
                i = pid_i * blk_n + ti
                j = pid_j * blk_h + tj
                if i < n and j < h:
                    o[i, m, j] = xl[ti, tj]
        if use_pdl:
            T.pdl_trigger()


@tilelang.jit
def expand_to_mhc_bwd_cuda(o_grad, x_grad, dtype: T.dtype = T.bfloat16, use_pdl: bool = False):
    assert dtype in (T.bfloat16, T.float32)
    n = T.dynamic('n')
    h, mhc = T.const('h, mhc')
    o_grad: T.Tensor[(n, mhc, h), dtype]
    x_grad: T.Tensor[(n, h), dtype]

    blk_n = 16
    blk_h = 128

    with T.Kernel(T.ceildiv(n, blk_n), T.ceildiv(h, blk_h)) as (pid_i, pid_j):
        if use_pdl:
            T.pdl_sync()
        xgl = T.alloc_fragment((blk_n, blk_h), T.float32)
        T.fill(xgl, 0)
        for m in T.serial(mhc):
            for ti, tj in T.Parallel(blk_n, blk_h):
                i = pid_i * blk_n + ti
                j = pid_j * blk_h + tj
                if i < n and j < h:
                    xgl[ti, tj] += o_grad[i, m, j]
        T.copy(xgl, x_grad[pid_i * blk_n, pid_j * blk_h])
        if use_pdl:
            T.pdl_trigger()

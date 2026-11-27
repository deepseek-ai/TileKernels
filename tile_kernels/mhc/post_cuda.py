import tilelang
from tilelang import language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def mhc_post_fwd_cuda(
    a,
    b,
    c,
    d,
    x,
    vec: int,
    n_thr: int = 128,
    h_blk: int = 1024,
    x_dtype: T.dtype = T.bfloat16,
    use_pdl: bool = False,
):
    assert x_dtype in (T.bfloat16, T.float32)
    n = T.dynamic('n')
    mhc = T.const('mhc')
    h = T.const('h')

    a: T.Tensor[(n, mhc, mhc), T.float32]
    b: T.Tensor[(n, mhc, h), x_dtype]
    c: T.Tensor[(n, mhc), T.float32]
    d: T.Tensor[(n, h), x_dtype]
    x: T.Tensor[(n, mhc, h), x_dtype]

    with T.Kernel(n, threads=n_thr) as pid_n:
        x_frag = T.alloc_fragment((mhc, h_blk), x_dtype)
        x_acc = T.alloc_fragment((mhc, h_blk), T.float32)
        b_shared = T.alloc_shared((mhc, h_blk), x_dtype)
        d_shared = T.alloc_shared(h_blk, x_dtype)

        a_frag = T.alloc_fragment((mhc, mhc), T.float32)
        c_frag = T.alloc_fragment(mhc, T.float32)
        T.copy(a[pid_n, 0, 0], a_frag)
        T.copy(c[pid_n, 0], c_frag)
        if use_pdl:
            T.pdl_sync()

        for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):
            T.copy(b[pid_n, 0, i0_h * h_blk], b_shared, disable_tma=True)
            T.copy(d[pid_n, i0_h * h_blk], d_shared, disable_tma=True)

            layout = T.Fragment(
                [mhc, h_blk],
                lambda i_mhco, i1_h: (i1_h // vec % n_thr, i_mhco * (h_blk // n_thr) + i1_h // vec // n_thr * vec + i1_h % vec),
            )
            T.annotate_layout({x_acc: layout, x_frag: layout})
            if x_dtype == T.bfloat16 and vec <= 2:
                for i_mhco, i1_h in T.Parallel(mhc, h_blk, loop_layout=layout):
                    x_local = T.alloc_var(T.float32)
                    x_local = T.fmul(c_frag[i_mhco], T.cast(d_shared[i1_h], T.float32))
                    for i_mhci in T.unroll(mhc):
                        x_local = T.fma(a_frag[i_mhci, i_mhco], T.cast(b_shared[i_mhci, i1_h], T.float32), x_local)
                    x_frag[i_mhco, i1_h] = x_local
            else:
                for i_mhco, i1_h in T.Parallel(mhc, h_blk, loop_layout=layout):
                    x_acc[i_mhco, i1_h] = T.fmul(c_frag[i_mhco], T.cast(d_shared[i1_h], T.float32))
                for i_mhci in T.unroll(mhc):
                    for i_mhco, i1_h in T.Parallel(mhc, h_blk, loop_layout=layout):
                        x_acc[i_mhco, i1_h] = T.fma(a_frag[i_mhci, i_mhco], T.cast(b_shared[i_mhci, i1_h], T.float32), x_acc[i_mhco, i1_h])
                T.copy(x_acc, x_frag)

            T.copy(x_frag, x[pid_n, 0, i0_h * h_blk], disable_tma=True)

        if use_pdl:
            T.pdl_trigger()


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def mhc_post_bwd_cuda(
    dx,
    a,
    b,
    c,
    d,
    n_thr: int = 128,
    h_blk: int = 256,
    x_dtype: T.dtype = T.bfloat16,
    use_pdl: bool = False,
):
    assert x_dtype in (T.bfloat16, T.float32)
    n = T.dynamic('n')
    h = T.const('h')

    dx: T.Tensor[(n, 4, h), x_dtype]
    a: T.Tensor[(n, 4, 4), T.float32]
    b: T.Tensor[(n, 4, h), x_dtype]
    c: T.Tensor[(n, 4), T.float32]
    d: T.Tensor[(n, h), x_dtype]

    da = T.empty((n, 4, 4), T.float32)
    db = T.empty((n, 4, h), x_dtype)
    dc = T.empty((n, 4), T.float32)
    dd = T.empty((n, h), x_dtype)

    with T.Kernel(n, threads=n_thr) as pid_n:
        if use_pdl:
            T.pdl_sync()

        dx_shared = T.alloc_shared((4, h_blk), x_dtype)
        b_shared = T.alloc_shared((4, h_blk), x_dtype)
        db_shared = T.alloc_shared((4, h_blk), x_dtype)
        d_shared = T.alloc_shared(h_blk, x_dtype)
        dd_shared = T.alloc_shared(h_blk, x_dtype)

        dx_local = T.alloc_fragment((4, h_blk), T.float32)
        b_local = T.alloc_fragment((4, h_blk), T.float32)
        db_local = T.alloc_fragment((4, h_blk), T.float32)
        d_local = T.alloc_fragment(h_blk, T.float32)
        dd_local = T.alloc_fragment(h_blk, T.float32)

        a_local = T.alloc_fragment((4, 4), T.float32)
        c_local = T.alloc_fragment(4, T.float32)
        T.copy(a[pid_n, 0, 0], a_local)
        T.copy(c[pid_n, 0], c_local)

        da_reducer = T.alloc_reducer((4, 4), T.float32, op='sum')
        dc_reducer = T.alloc_reducer(4, T.float32, op='sum')
        da_result = T.alloc_fragment((4, 4), T.float32)
        dc_result = T.alloc_fragment(4, T.float32)
        T.reducer_init(da_reducer)
        T.reducer_init(dc_reducer)

        for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=3):
            T.copy(dx[pid_n, 0, i0_h * h_blk], dx_shared, disable_tma=True)
            T.copy(b[pid_n, 0, i0_h * h_blk], b_shared, disable_tma=True)
            T.copy(d[pid_n, i0_h * h_blk], d_shared, disable_tma=True)

            T.copy(dx_shared, dx_local)
            T.copy(b_shared, b_local)
            T.copy(d_shared, d_local)

            # da and db
            T.clear(db_local)
            for i_mhci in T.serial(4):
                for i_mhco in T.serial(4):
                    for i1_h in T.Parallel(h_blk):
                        db_local[i_mhci, i1_h] += a_local[i_mhci, i_mhco] * dx_local[i_mhco, i1_h]
                        T.reducer_update(da_reducer[i_mhci, i_mhco], b_local[i_mhci, i1_h] * dx_local[i_mhco, i1_h])

            # dc and dd
            T.clear(dd_local)
            for i_mhc in T.serial(4):
                for i1_h in T.Parallel(h_blk):
                    T.reducer_update(dc_reducer[i_mhc], d_local[i1_h] * dx_local[i_mhc, i1_h])
                    dd_local[i1_h] += c_local[i_mhc] * dx_local[i_mhc, i1_h]

            T.copy(db_local, db_shared)
            T.copy(dd_local, dd_shared)

            T.copy(db_shared, db[pid_n, 0, i0_h * h_blk], disable_tma=True)
            T.copy(dd_shared, dd[pid_n, i0_h * h_blk], disable_tma=True)

        T.finalize_reducer(da_reducer, da_result, batch=16)
        T.finalize_reducer(dc_reducer, dc_result, batch=4)
        T.copy(da_result, da[pid_n, 0, 0])
        T.copy(dc_result, dc[pid_n, 0])

        if use_pdl:
            T.pdl_trigger()

    return da, db, dc, dd

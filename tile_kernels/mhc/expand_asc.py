import tilelang
from tilelang.ascend import language as T

from tile_kernels.mhc.common_asc import _VEC, _VREG_BYTES, _load_f32, _store_f32


@tilelang.jit
def expand_to_mhc_fwd_asc(x, o, h_blk: int, num_stages: int, num_vec_cores: int, x_dtype: T.dtype = T.bfloat16):
    assert x_dtype in (T.bfloat16, T.float32)
    # Expand copies raw bits: one 256B vector register, i.e. 128 bf16 or 64 fp32 elements.
    chunk = _VREG_BYTES // x_dtype.bytes
    num_tokens = T.dynamic('num_tokens')
    h, mhc = T.const('h, mhc')
    x: T.Tensor[(num_tokens, h), x_dtype]
    o: T.Tensor[(num_tokens, mhc, h), x_dtype]

    h_blk = T.min(h_blk, h)

    with T.Kernel(num_vec_cores) as core_id:
        x_ub = T.alloc_shared((h_blk,), x_dtype)
        o_ub = T.alloc_shared((mhc, h_blk), x_dtype)
        T.annotate_buffer_versions({x_ub: num_stages, o_ub: num_stages})

        for pid_n, pid_h in T.Persistent(
            [num_tokens, T.ceildiv(h, h_blk)],
            num_vec_cores,
            core_id,
            group_size=1,
            num_stages=num_stages,
        ):
            T.copy(x[pid_n, pid_h * h_blk], x_ub)
            with T.SimdVF():
                mask = T.simd.pset(x_dtype.bits)
                num_chunks = h_blk // chunk
                for c in T.serial(num_chunks):
                    col = c * chunk
                    xb = T.simd.vld(x_ub[col])
                    for m in T.unroll(mhc, explicit=True):
                        T.simd.vsts(o_ub[m, col], xb, mask)

            T.copy(o_ub, o[pid_n, 0, pid_h * h_blk])


@tilelang.jit
def expand_to_mhc_bwd_asc(o_grad, x_grad, h_blk: int, num_stages: int, num_vec_cores: int, x_dtype: T.dtype = T.bfloat16):
    assert x_dtype in (T.bfloat16, T.float32)
    num_tokens = T.dynamic('num_tokens')
    h, mhc = T.const('h, mhc')
    o_grad: T.Tensor[(num_tokens, mhc, h), x_dtype]
    x_grad: T.Tensor[(num_tokens, h), x_dtype]

    h_blk = T.min(h_blk, h)

    with T.Kernel(num_vec_cores) as core_id:
        o_grad_ub = T.alloc_shared((mhc, h_blk), x_dtype)
        x_grad_ub = T.alloc_shared((h_blk,), x_dtype)
        T.annotate_buffer_versions({o_grad_ub: num_stages, x_grad_ub: num_stages})

        for pid_n, pid_h in T.Persistent(
            [num_tokens, T.ceildiv(h, h_blk)],
            num_vec_cores,
            core_id,
            group_size=1,
            num_stages=num_stages,
        ):
            T.copy(o_grad[pid_n, 0, pid_h * h_blk], o_grad_ub)

            with T.SimdVF():
                mask_32b = T.simd.pset(32)
                acc = T.simd.alloc_var(T.float32)
                num_chunks = h_blk // _VEC
                for c in range(num_chunks):
                    col = c * _VEC
                    acc = _load_f32(o_grad_ub[0, col])
                    for m in T.unroll(mhc - 1, explicit=True):
                        acc = T.simd.vadd(acc, _load_f32(o_grad_ub[m + 1, col]), mask_32b)
                    _store_f32(x_grad_ub[col], acc)

            T.copy(x_grad_ub, x_grad[pid_n, pid_h * h_blk])

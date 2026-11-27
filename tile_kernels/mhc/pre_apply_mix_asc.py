import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from tile_kernels.mhc.common_asc import _VEC, _load_f32, _store_f32


@tilelang.jit()
def mhc_pre_apply_mix_fwd_asc(x, mix, o, h_blk, num_vec_cores: int, x_dtype: T.dtype = T.bfloat16):
    """Ascend forward: ``o[n, h] = sum_mhc mix[n, mhc] * x[n, mhc, h]``."""
    assert x_dtype in (T.bfloat16, T.float32)
    num_tokens = T.dynamic('num_tokens')
    mhc = T.const('mhc')
    hidden = T.const('hidden')
    x: T.Tensor[(num_tokens, mhc, hidden), x_dtype]
    mix: T.Tensor[(num_tokens, mhc), T.float32]
    o: T.Tensor[(num_tokens, hidden), x_dtype]

    num_hidden_tiles = T.ceildiv(hidden, h_blk)
    num_chunks = h_blk // _VEC
    num_stages = 3

    with T.Kernel(num_vec_cores) as core_id:
        x_ub = T.alloc_shared((mhc, h_blk), x_dtype)
        mix_ub = T.alloc_shared((T.ceildiv(mhc, 8) * 8,), T.float32)  # TODO remove manual padding
        o_ub = T.alloc_shared((h_blk,), x_dtype)
        T.annotate_buffer_versions({x_ub: num_stages, mix_ub: num_stages, o_ub: num_stages})

        for pid_n, pid_h in T.Persistent([num_tokens, num_hidden_tiles], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            T.copy(mix[pid_n, 0], mix_ub[0:mhc])
            T.copy(x[pid_n, 0, pid_h * h_blk], x_ub)

            with T.SimdVF():
                # Preload the mix scalars as broadcast registers once per token-tile.
                mix_reg = S.alloc_local((mhc,), T.float32)
                for m in T.unroll(mhc, explicit=True):
                    mix_reg[m] = S.vld(mix_ub[m], dist='BRC_B32')
                xs = S.alloc_local((mhc,), T.float32)
                acc = S.alloc_var(T.float32)
                for c in T.serial(num_chunks):
                    col = c * _VEC
                    for m in T.unroll(mhc, explicit=True):
                        xs[m] = _load_f32(x_ub[m, col])
                    acc = S.vmul(xs[0], mix_reg[0])
                    for m in T.unroll(mhc - 1, explicit=True):
                        S.vmula(acc, xs[m + 1], mix_reg[m + 1])
                    _store_f32(o_ub[col], acc)

            T.copy(o_ub, o[pid_n, pid_h * h_blk])


@tilelang.jit()
def mhc_pre_apply_mix_bwd_asc(o_grad, x, mix, x_grad, h_blk, num_stages, num_vec_cores: int, x_dtype: T.dtype = T.bfloat16):
    """Ascend backward.

    ``x_grad[n, m, h] += mix[n, m] * o_grad[n, h]``  (accumulated onto existing values)
    ``mix_grad[n, m] = sum_h o_grad[n, h] * x[n, m, h]``  (reduction over h)
    """
    assert x_dtype in (T.bfloat16, T.float32)
    num_tokens = T.dynamic('num_tokens')
    mhc = T.const('mhc')
    hidden = T.const('hidden')
    o_grad: T.Tensor[(num_tokens, hidden), x_dtype]
    x: T.Tensor[(num_tokens, mhc, hidden), x_dtype]
    mix: T.Tensor[(num_tokens, mhc), T.float32]
    x_grad: T.Tensor[(num_tokens, mhc, hidden), x_dtype]
    mix_grad = T.empty((num_tokens, mhc), T.float32)

    num_hidden_tiles = T.ceildiv(hidden, h_blk)
    num_chunks = h_blk // _VEC

    with T.Kernel(num_vec_cores) as core_id:
        o_grad_ub = T.alloc_shared((h_blk,), x_dtype)
        x_ub = T.alloc_shared((mhc, h_blk), x_dtype)
        x_grad_ub = T.alloc_shared((mhc, h_blk), x_dtype)
        mix_ub = T.alloc_shared((T.ceildiv(mhc, 8) * 8,), T.float32)  # TODO remove manual padding
        mix_grad_acc = T.alloc_shared((mhc, _VEC), T.float32)
        mix_grad_ub = T.alloc_shared((T.ceildiv(mhc, 8) * 8,), T.float32)  # TODO remove manual padding
        T.annotate_buffer_versions(
            {
                o_grad_ub: num_stages,
                x_ub: num_stages,
                x_grad_ub: num_stages,
                mix_ub: num_stages,
                mix_grad_ub: num_stages,
            }
        )

        for pid_n in T.Persistent([num_tokens], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            T.copy(mix[pid_n, 0], mix_ub[0:mhc])

            with T.SimdVF():
                zero = S.vdup(0.0, T.float32)
                for m in T.unroll(mhc, explicit=True):
                    S.vsts(mix_grad_acc[m, 0], zero)

            for pid_h in T.serial(num_hidden_tiles):
                T.copy(o_grad[pid_n, pid_h * h_blk], o_grad_ub)
                T.copy(x[pid_n, 0, pid_h * h_blk], x_ub)
                T.copy(x_grad[pid_n, 0, pid_h * h_blk], x_grad_ub)

                with T.SimdVF():
                    mix_reg = S.alloc_local((mhc,), T.float32)
                    for m in T.unroll(mhc, explicit=True):
                        mix_reg[m] = S.vld(mix_ub[m], dist='BRC_B32')
                    mix_grad_reg = S.alloc_var(T.float32)
                    x_grad_f32 = S.alloc_var(T.float32)
                    for m in T.unroll(mhc, explicit=True):
                        mix_grad_reg = S.vld(mix_grad_acc[m, 0])
                        for c in T.serial(num_chunks):
                            col = c * _VEC
                            o_grad_f32 = _load_f32(o_grad_ub[col])
                            x_f32 = _load_f32(x_ub[m, col])
                            x_grad_f32 = _load_f32(x_grad_ub[m, col])
                            S.vmula(x_grad_f32, o_grad_f32, mix_reg[m])
                            _store_f32(x_grad_ub[m, col], x_grad_f32)
                            S.vmula(mix_grad_reg, o_grad_f32, x_f32)
                        S.vsts(mix_grad_acc[m, 0], mix_grad_reg)

                T.copy(x_grad_ub, x_grad[pid_n, 0, pid_h * h_blk])

            with T.SimdVF():
                for m in T.unroll(mhc, explicit=True):
                    S.vsts(mix_grad_ub[m], S.vcadd(S.vld(mix_grad_acc[m, 0])), dist='ONEPT_B32')
            T.copy(mix_grad_ub[0:mhc], mix_grad[pid_n, 0])

    return mix_grad

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from tile_kernels.mhc.common_asc import _VEC, _load_f32, _store_f32


@tilelang.jit()
def mhc_post_fwd_asc(
    comb_res_mix,
    residual,
    post_layer_mix,
    x,
    out,
    h_blk,
    num_vec_cores: int,
    x_dtype: T.dtype = T.bfloat16,
):
    """Ascend forward.

    ``out[n, mo, h] = post_layer_mix[n, mo] * x[n, h]
                      + sum_mi comb_res_mix[n, mi, mo] * residual[n, mi, h]``
    """
    assert x_dtype in (T.bfloat16, T.float32), f'{x_dtype=}'
    num_tokens = T.dynamic('num_tokens')
    mhc = T.const('mhc')
    hidden = T.const('hidden')
    comb_res_mix: T.Tensor[(num_tokens, mhc, mhc), T.float32]
    residual: T.Tensor[(num_tokens, mhc, hidden), x_dtype]
    post_layer_mix: T.Tensor[(num_tokens, mhc), T.float32]
    x: T.Tensor[(num_tokens, hidden), x_dtype]
    out: T.Tensor[(num_tokens, mhc, hidden), x_dtype]

    num_hidden_tiles = T.ceildiv(hidden, h_blk)
    num_chunks = h_blk // _VEC
    num_stages = 3

    with T.Kernel(num_vec_cores) as core_id:
        residual_ub = T.alloc_shared((mhc, h_blk), x_dtype)
        x_ub = T.alloc_shared((h_blk,), x_dtype)
        comb_res_mix_ub = T.alloc_shared((mhc, mhc), T.float32)
        post_layer_mix_ub = T.alloc_shared((T.ceildiv(mhc, 8) * 8,), T.float32)  # TODO remove manual padding
        out_ub = T.alloc_shared((mhc, h_blk), x_dtype)
        T.annotate_buffer_versions(
            {
                residual_ub: num_stages,
                x_ub: num_stages,
                comb_res_mix_ub: num_stages,
                post_layer_mix_ub: num_stages,
                out_ub: num_stages,
            }
        )

        for pid_n, pid_h in T.Persistent([num_tokens, num_hidden_tiles], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            T.copy(comb_res_mix[pid_n, 0, 0], comb_res_mix_ub)
            T.copy(post_layer_mix[pid_n, 0], post_layer_mix_ub[0:mhc])
            T.copy(residual[pid_n, 0, pid_h * h_blk], residual_ub)
            T.copy(x[pid_n, pid_h * h_blk], x_ub)

            with T.SimdVF():
                comb_mix = S.alloc_local((mhc * mhc,), T.float32)
                post_mix = S.alloc_local((mhc,), T.float32)
                for mi in T.unroll(mhc, explicit=True):
                    for mo in T.unroll(mhc, explicit=True):
                        comb_mix[mi * mhc + mo] = S.vld(comb_res_mix_ub[mi, mo], dist='BRC_B32')
                for mo in T.unroll(mhc, explicit=True):
                    post_mix[mo] = S.vld(post_layer_mix_ub[mo], dist='BRC_B32')
                out_reg = S.alloc_local((mhc,), T.float32)
                res = S.alloc_local((mhc,), T.float32)
                for c in T.serial(num_chunks):
                    col = c * _VEC
                    x_f32 = _load_f32(x_ub[col])
                    for mi in T.unroll(mhc, explicit=True):
                        res[mi] = _load_f32(residual_ub[mi, col])
                    for mo in T.unroll(mhc, explicit=True):
                        out_reg[mo] = S.vmul(x_f32, post_mix[mo])
                    for mi in T.unroll(mhc, explicit=True):
                        for mo in T.unroll(mhc, explicit=True):
                            S.vmula(out_reg[mo], res[mi], comb_mix[mi * mhc + mo])
                    for mo in T.unroll(mhc, explicit=True):
                        _store_f32(out_ub[mo, col], out_reg[mo])

            T.copy(out_ub, out[pid_n, 0, pid_h * h_blk])


@tilelang.jit()
def mhc_post_bwd_asc(d_o, comb_res_mix, residual, post_layer_mix, x, h_blk, num_stages, num_vec_cores: int, x_dtype: T.dtype = T.bfloat16):
    """Ascend backward.

    d_x[n, h] = sum_mo d_o[n, mo, h] * post_layer_mix[n, mo]
    d_post_layer_mix[n, mo] = sum_h d_o[n, mo, h] * x[n, h]
    d_residual[n, mi, h] = sum_mo d_o[n, mo, h] * comb_res_mix[n, mi, mo]
    d_comb_res_mix[n, mi, mo] = sum_h d_o[n, mo, h] * residual[n, mi, h]
    """
    assert x_dtype in (T.bfloat16, T.float32), f'{x_dtype=}'
    num_tokens = T.dynamic('num_tokens')
    mhc = T.const('mhc')
    hidden = T.const('hidden')
    d_o: T.Tensor[(num_tokens, mhc, hidden), x_dtype]
    comb_res_mix: T.Tensor[(num_tokens, mhc, mhc), T.float32]
    residual: T.Tensor[(num_tokens, mhc, hidden), x_dtype]
    post_layer_mix: T.Tensor[(num_tokens, mhc), T.float32]
    x: T.Tensor[(num_tokens, hidden), x_dtype]

    d_comb_res_mix = T.empty((num_tokens, mhc, mhc), T.float32)
    d_residual = T.empty((num_tokens, mhc, hidden), x_dtype)
    d_post_layer_mix = T.empty((num_tokens, mhc), T.float32)
    d_x = T.empty((num_tokens, hidden), x_dtype)

    num_hidden_tiles = T.ceildiv(hidden, h_blk)
    num_chunks = h_blk // _VEC

    with T.Kernel(num_vec_cores) as core_id:
        d_o_ub = T.alloc_shared((mhc, h_blk), x_dtype)
        x_ub = T.alloc_shared((h_blk,), x_dtype)
        residual_ub = T.alloc_shared((mhc, h_blk), x_dtype)
        comb_res_mix_ub = T.alloc_shared((mhc, mhc), T.float32)
        post_layer_mix_ub = T.alloc_shared((T.ceildiv(mhc, 8) * 8,), T.float32)  # TODO remove manual padding
        d_x_ub = T.alloc_shared((h_blk,), x_dtype)
        d_residual_ub = T.alloc_shared((mhc, h_blk), x_dtype)
        d_post_mix_ub = T.alloc_shared((T.ceildiv(mhc, 8) * 8,), T.float32)  # TODO remove manual padding
        d_comb_mix_ub = T.alloc_shared((mhc, mhc), T.float32)
        post_mix_acc = T.alloc_shared((mhc, _VEC), T.float32)
        comb_mix_acc = T.alloc_shared((mhc * mhc, _VEC), T.float32)
        T.annotate_buffer_versions(
            {
                d_o_ub: num_stages,
                x_ub: num_stages,
                residual_ub: num_stages,
                d_x_ub: num_stages,
                d_residual_ub: num_stages,
                comb_res_mix_ub: num_stages,
                post_layer_mix_ub: num_stages,
                d_post_mix_ub: 1,
                d_comb_mix_ub: 1,
                post_mix_acc: 1,
                comb_mix_acc: 1,
            }
        )

        for pid_n in T.Persistent([num_tokens], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            T.copy(comb_res_mix[pid_n, 0, 0], comb_res_mix_ub)
            T.copy(post_layer_mix[pid_n, 0], post_layer_mix_ub[0:mhc])

            with T.SimdVF():
                zero = S.vdup(0.0, T.float32)
                for mo in T.unroll(mhc, explicit=True):
                    S.vsts(post_mix_acc[mo, 0], zero)
                for idx in T.unroll(mhc * mhc, explicit=True):
                    S.vsts(comb_mix_acc[idx, 0], zero)

            for pid_h in T.serial(num_hidden_tiles):
                T.copy(d_o[pid_n, 0, pid_h * h_blk], d_o_ub)
                T.copy(x[pid_n, pid_h * h_blk], x_ub)
                T.copy(residual[pid_n, 0, pid_h * h_blk], residual_ub)

                with T.SimdVF():
                    post_mix = S.alloc_local((mhc,), T.float32)
                    for mo in T.unroll(mhc, explicit=True):
                        post_mix[mo] = S.vld(post_layer_mix_ub[mo], dist='BRC_B32')
                    comb_mix = S.alloc_local((mhc * mhc,), T.float32)
                    for mi in T.unroll(mhc, explicit=True):
                        for mo in T.unroll(mhc, explicit=True):
                            comb_mix[mi * mhc + mo] = S.vld(comb_res_mix_ub[mi, mo], dist='BRC_B32')

                    d_o_reg = S.alloc_local((mhc,), T.float32)
                    d_x_reg = S.alloc_var(T.float32)
                    d_res_reg = S.alloc_var(T.float32)
                    post_acc_reg = S.alloc_var(T.float32)
                    d_comb_mix_mi = S.alloc_local((mhc,), T.float32)

                    for c in T.serial(num_chunks):
                        col = c * _VEC
                        x_f32 = _load_f32(x_ub[col])
                        for mo in T.unroll(mhc, explicit=True):
                            d_o_reg[mo] = _load_f32(d_o_ub[mo, col])

                        d_x_reg = S.vdup(0.0, T.float32)
                        for mo in T.unroll(mhc, explicit=True):
                            S.vmula(d_x_reg, d_o_reg[mo], post_mix[mo])
                        _store_f32(d_x_ub[col], d_x_reg)

                        for mo in T.unroll(mhc, explicit=True):
                            post_acc_reg = S.vld(post_mix_acc[mo, 0])
                            S.vmula(post_acc_reg, d_o_reg[mo], x_f32)
                            S.vsts(post_mix_acc[mo, 0], post_acc_reg)

                        for mi in T.unroll(mhc, explicit=True):
                            res_mi = _load_f32(residual_ub[mi, col])
                            for mo in T.unroll(mhc, explicit=True):
                                d_comb_mix_mi[mo] = S.vld(comb_mix_acc[mi * mhc + mo, 0])
                            d_res_reg = S.vdup(0.0, T.float32)
                            for mo in T.unroll(mhc, explicit=True):
                                S.vmula(d_res_reg, d_o_reg[mo], comb_mix[mi * mhc + mo])
                                S.vmula(d_comb_mix_mi[mo], d_o_reg[mo], res_mi)
                            _store_f32(d_residual_ub[mi, col], d_res_reg)
                            for mo in T.unroll(mhc, explicit=True):
                                S.vsts(comb_mix_acc[mi * mhc + mo, 0], d_comb_mix_mi[mo])

                T.copy(d_x_ub, d_x[pid_n, pid_h * h_blk])
                T.copy(d_residual_ub, d_residual[pid_n, 0, pid_h * h_blk])

            with T.SimdVF():
                for mo in T.unroll(mhc, explicit=True):
                    S.vsts(d_post_mix_ub[mo], S.vcadd(S.vld(post_mix_acc[mo, 0])), dist='ONEPT_B32')
                for mi in T.unroll(mhc, explicit=True):
                    for mo in T.unroll(mhc, explicit=True):
                        S.vsts(d_comb_mix_ub[mi, mo], S.vcadd(S.vld(comb_mix_acc[mi * mhc + mo, 0])), dist='ONEPT_B32')

            T.copy(d_post_mix_ub[0:mhc], d_post_layer_mix[pid_n, 0])
            T.copy(d_comb_mix_ub, d_comb_res_mix[pid_n, 0, 0])

    return d_comb_res_mix, d_residual, d_post_layer_mix, d_x

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S


def _u32_index(index_i32):
    return T.reinterpret(index_i32, 'uint32x64')


@tilelang.jit
def mhc_pre_split_mixes_fwd_asc(
    input_mixes,
    mhc_scale,
    mhc_base,
    pre_layer_mix,
    post_layer_mix,
    comb_res_mix,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
    token_block_size: int,
    num_stages: int,
    dtype: T.dtype,
    num_vec_cores: int,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult = T.const('mhc_mult')
    mhc_mult2 = T.const('mhc_mult2')
    mhc_mult3 = T.const('mhc_mult3')

    input_mixes: T.Tensor[(num_tokens, mhc_mult3), dtype]
    mhc_scale: T.Tensor[(3,), dtype]
    mhc_base: T.Tensor[mhc_mult3, dtype]
    pre_layer_mix: T.Tensor[(num_tokens, mhc_mult), dtype]
    post_layer_mix: T.Tensor[(num_tokens, mhc_mult), dtype]
    comb_res_mix: T.Tensor[(num_tokens, mhc_mult2), dtype]

    with T.Kernel(num_vec_cores) as core_id:
        for pid in T.Persistent([T.ceildiv(num_tokens, token_block_size)], num_vec_cores, core_id, num_stages=num_stages):
            input_mixes_ub = T.alloc_shared((token_block_size, mhc_mult3), dtype)
            pre_layer_mix_ub = T.alloc_shared((token_block_size, mhc_mult), dtype)
            post_layer_mix_ub = T.alloc_shared((token_block_size, mhc_mult), dtype)
            comb_res_mix_ub = T.alloc_shared((token_block_size, mhc_mult2), dtype)
            T.copy(input_mixes[pid * token_block_size, 0], input_mixes_ub)

            with T.SimtVF(threads=256):
                for i1, j in T.Parallel(token_block_size, mhc_mult):
                    i = pid * token_block_size + i1
                    if i < num_tokens:
                        pre_layer_mix_ub[i1, j] = T.sigmoid(input_mixes_ub[i1, j] * mhc_scale[0] + mhc_base[j]) + mhc_pre_eps

                for i1, j in T.Parallel(token_block_size, mhc_mult):
                    i = pid * token_block_size + i1
                    if i < num_tokens:
                        post_layer_mix_ub[i1, j] = (
                            T.sigmoid(input_mixes_ub[i1, j + mhc_mult] * mhc_scale[1] + mhc_base[j + mhc_mult]) * mhc_post_mult_value
                        )

                for i1, j in T.Parallel(token_block_size, mhc_mult):
                    i = pid * token_block_size + i1
                    if i < num_tokens:
                        comb_res_mix_ub[i1, j] = input_mixes_ub[i1, j + mhc_mult * 2] * mhc_scale[2] + mhc_base[j + mhc_mult * 2]

                for i1, j in T.Parallel(token_block_size, mhc_mult):
                    i = pid * token_block_size + i1
                    if i < num_tokens:
                        comb_res_mix_ub[i1, j + mhc_mult] = (
                            input_mixes_ub[i1, j + mhc_mult * 2 + mhc_mult] * mhc_scale[2] + mhc_base[j + mhc_mult * 2 + mhc_mult]
                        )

                for i1, j in T.Parallel(token_block_size, mhc_mult):
                    i = pid * token_block_size + i1
                    if i < num_tokens:
                        comb_res_mix_ub[i1, j + mhc_mult * 2] = (
                            input_mixes_ub[i1, j + mhc_mult * 2 + mhc_mult * 2] * mhc_scale[2] + mhc_base[j + mhc_mult * 2 + mhc_mult * 2]
                        )

                for i1, j in T.Parallel(token_block_size, mhc_mult):
                    i = pid * token_block_size + i1
                    if i < num_tokens:
                        comb_res_mix_ub[i1, j + mhc_mult * 3] = (
                            input_mixes_ub[i1, j + mhc_mult * 2 + mhc_mult * 3] * mhc_scale[2] + mhc_base[j + mhc_mult * 2 + mhc_mult * 3]
                        )

            T.copy(pre_layer_mix_ub, pre_layer_mix[pid * token_block_size, 0])
            T.copy(post_layer_mix_ub, post_layer_mix[pid * token_block_size, 0])
            T.copy(comb_res_mix_ub, comb_res_mix[pid * token_block_size, 0])


@tilelang.jit
def mhc_pre_split_mixes_fwd_simd_mhc4_asc(
    input_mixes,
    mhc_scale,
    mhc_base,
    pre_layer_mix,
    post_layer_mix,
    comb_res_mix,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
    token_block_size: int,
    num_stages: int,
    num_vec_cores: int,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult = 4
    mhc_mult2 = 16
    mhc_mult3 = 24

    input_mixes: T.Tensor[(num_tokens, mhc_mult3), T.float32]
    mhc_scale: T.Tensor[(3,), T.float32]
    mhc_base: T.Tensor[mhc_mult3, T.float32]
    pre_layer_mix: T.Tensor[(num_tokens, mhc_mult), T.float32]
    post_layer_mix: T.Tensor[(num_tokens, mhc_mult), T.float32]
    comb_res_mix: T.Tensor[(num_tokens, mhc_mult2), T.float32]

    with T.Kernel(num_vec_cores) as core_id:
        scale_ub = T.alloc_shared((3, 8), T.float32)
        base_ub = T.alloc_shared((mhc_mult3, 8), T.float32)

        with T.SimtVF(threads=32):
            for i in T.Parallel(3):
                scale_ub[i, 0] = mhc_scale[i]
            for i in T.Parallel(mhc_mult3):
                base_ub[i, 0] = mhc_base[i]

        input_mixes_ub = T.alloc_shared((token_block_size, mhc_mult3), T.float32)
        pre_layer_mix_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        post_layer_mix_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        comb_res_mix_ub = T.alloc_shared((token_block_size, mhc_mult2), T.float32)
        T.annotate_buffer_versions(
            {
                input_mixes_ub: num_stages,
                pre_layer_mix_ub: num_stages,
                post_layer_mix_ub: num_stages,
                comb_res_mix_ub: num_stages,
            }
        )

        for pid in T.Persistent([T.ceildiv(num_tokens, token_block_size)], num_vec_cores, core_id, num_stages=num_stages):
            T.copy(input_mixes[pid * token_block_size, 0], input_mixes_ub)

            with T.SimdVF():
                mask = S.pset(32)
                idx = S.vci(T.int32(0), T.int32)
                zeros = S.vdup(T.float32(0), T.float32, mask)
                ones = S.vdup(T.float32(1), T.float32, mask)
                col3 = S.vand(idx, S.vdup(T.int32(3), T.int32, mask), mask)
                col15 = S.vand(idx, S.vdup(T.int32(15), T.int32, mask), mask)
                scale0 = S.vld(scale_ub[0, 0], 'BRC_B32')
                scale1 = S.vld(scale_ub[1, 0], 'BRC_B32')
                scale2 = S.vld(scale_ub[2, 0], 'BRC_B32')
                pre_base_vec = S.alloc_var(T.float32)
                post_base_vec = S.alloc_var(T.float32)
                comb_base_vec = S.alloc_var(T.float32)
                pre_base_vec = zeros
                post_base_vec = zeros
                comb_base_vec = zeros

                for base_id in range(4):
                    base_mask = S.vcmps(col3, T.int32(base_id), mask, 'eq')
                    pre_base_vec = S.vsel(S.vld(base_ub[base_id, 0], 'BRC_B32'), pre_base_vec, base_mask)
                    post_base_vec = S.vsel(S.vld(base_ub[base_id + 4, 0], 'BRC_B32'), post_base_vec, base_mask)

                for base_id in range(16):
                    base_mask = S.vcmps(col15, T.int32(base_id), mask, 'eq')
                    comb_base_vec = S.vsel(S.vld(base_ub[base_id + 8, 0], 'BRC_B32'), comb_base_vec, base_mask)

                for vec_id in T.serial(4):
                    token = S.vadd(S.vshrs(idx, T.int32(2), mask), S.vdup(T.int32(vec_id * 16), T.int32, mask), mask)
                    token_offset = S.vadd(S.vshls(token, T.int32(4), mask), S.vshls(token, T.int32(3), mask), mask)
                    pre_index = S.vadd(token_offset, col3, mask)
                    post_index = S.vadds(pre_index, T.int32(4), mask)

                    pre_val = S.vadd(S.vmul(S.vgather2(input_mixes_ub[0, 0], _u32_index(pre_index), mask), scale0, mask), pre_base_vec, mask)
                    pre_sig = S.vdiv(ones, S.vadds(S.vexpdif(zeros, pre_val, mask), T.float32(1), mask), mask)
                    S.vsts(pre_layer_mix_ub[vec_id * 16, 0], S.vadds(pre_sig, T.float32(mhc_pre_eps), mask), mask)

                    post_val = S.vadd(S.vmul(S.vgather2(input_mixes_ub[0, 0], _u32_index(post_index), mask), scale1, mask), post_base_vec, mask)
                    post_sig = S.vdiv(ones, S.vadds(S.vexpdif(zeros, post_val, mask), T.float32(1), mask), mask)
                    S.vsts(post_layer_mix_ub[vec_id * 16, 0], S.vmuls(post_sig, T.float32(mhc_post_mult_value), mask), mask)

                for vec_id in T.serial(16):
                    token = S.vadd(S.vshrs(idx, T.int32(4), mask), S.vdup(T.int32(vec_id * 4), T.int32, mask), mask)
                    token_offset = S.vadd(S.vshls(token, T.int32(4), mask), S.vshls(token, T.int32(3), mask), mask)
                    comb_index = S.vadd(S.vadds(token_offset, T.int32(8), mask), col15, mask)
                    comb_val = S.vadd(
                        S.vmul(S.vgather2(input_mixes_ub[0, 0], _u32_index(comb_index), mask), scale2, mask),
                        comb_base_vec,
                        mask,
                    )
                    S.vsts(comb_res_mix_ub[vec_id * 4, 0], comb_val, mask)

            T.copy(pre_layer_mix_ub, pre_layer_mix[pid * token_block_size, 0])
            T.copy(post_layer_mix_ub, post_layer_mix[pid * token_block_size, 0])
            T.copy(comb_res_mix_ub, comb_res_mix[pid * token_block_size, 0])


@tilelang.jit
def mhc_pre_split_mixes_bwd_asc(
    pre_layer_mix_grad,
    post_layer_mix_grad,
    comb_res_mix_grad,
    input_mixes,
    post_layer_mix,
    mhc_scale,
    mhc_base,
    input_mixes_grad,
    mhc_scale_grad_partial,
    mhc_base_grad_partial,
    mhc_post_mult_value: float,
    token_block_size: int,
    dtype: T.dtype,
    num_vec_cores: int,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult = T.const('mhc_mult')
    mhc_mult2 = T.const('mhc_mult2')
    mhc_mult3 = T.const('mhc_mult3')

    pre_layer_mix_grad: T.Tensor[(num_tokens, mhc_mult), dtype]
    post_layer_mix_grad: T.Tensor[(num_tokens, mhc_mult), dtype]
    comb_res_mix_grad: T.Tensor[(num_tokens, mhc_mult2), dtype]
    input_mixes: T.Tensor[(num_tokens, mhc_mult3), dtype]
    post_layer_mix: T.Tensor[(num_tokens, mhc_mult), dtype]
    mhc_scale: T.Tensor[(3,), dtype]
    mhc_base: T.Tensor[mhc_mult3, dtype]
    input_mixes_grad: T.Tensor[(num_tokens, mhc_mult3), dtype]
    mhc_scale_grad_partial: T.Tensor[(num_vec_cores, 3), dtype]
    mhc_base_grad_partial: T.Tensor[(num_vec_cores, mhc_mult3), dtype]

    with T.Kernel(num_vec_cores) as pid:
        with T.SimtVF(threads=256):
            scale_grad_0 = T.alloc_reducer(1, dtype, op='sum')
            scale_grad_1 = T.alloc_reducer(1, dtype, op='sum')
            scale_grad_2 = T.alloc_reducer(1, dtype, op='sum')
            base_grad_reducer = T.alloc_reducer(mhc_mult3, dtype, op='sum')
            scale_grad_0_frag = T.alloc_fragment(1, dtype)
            scale_grad_1_frag = T.alloc_fragment(1, dtype)
            scale_grad_2_frag = T.alloc_fragment(1, dtype)
            base_grad_frag = T.alloc_fragment(mhc_mult3, dtype)
            T.reducer_init(scale_grad_0)
            T.reducer_init(scale_grad_1)
            T.reducer_init(scale_grad_2)
            T.reducer_init(base_grad_reducer)

            for t in T.Persistent([T.ceildiv(num_tokens, token_block_size)], num_vec_cores, pid, group_size=1):
                grad_frag = T.alloc_fragment((token_block_size, mhc_mult3), dtype)

                for i1, j in T.Parallel(token_block_size, mhc_mult3):
                    i = t * token_block_size + i1
                    if i < num_tokens:
                        inp = input_mixes[i, j]
                        if j < mhc_mult:
                            pre_act = T.sigmoid(inp * mhc_scale[0] + mhc_base[j])
                            grad_frag[i1, j] = pre_layer_mix_grad[i, j] * pre_act * (1 - pre_act)
                        if j >= mhc_mult and j < mhc_mult * 2:
                            post_act = post_layer_mix[i, j - mhc_mult]
                            grad_frag[i1, j] = post_layer_mix_grad[i, j - mhc_mult] * post_act * (1 - post_act / mhc_post_mult_value)
                        if j >= mhc_mult * 2:
                            grad_frag[i1, j] = comb_res_mix_grad[i, j - mhc_mult * 2]

                for i1, j in T.Parallel(token_block_size, mhc_mult3):
                    i = t * token_block_size + i1
                    if i < num_tokens:
                        inp = input_mixes[i, j]
                        if j < mhc_mult:
                            T.reducer_update(scale_grad_0[0], grad_frag[i1, j] * inp)
                            T.reducer_update(base_grad_reducer[j], grad_frag[i1, j])
                            input_mixes_grad[i, j] = grad_frag[i1, j] * mhc_scale[0]
                        if j >= mhc_mult and j < mhc_mult * 2:
                            T.reducer_update(scale_grad_1[0], grad_frag[i1, j] * inp)
                            T.reducer_update(base_grad_reducer[j], grad_frag[i1, j])
                            input_mixes_grad[i, j] = grad_frag[i1, j] * mhc_scale[1]
                        if j >= mhc_mult * 2:
                            T.reducer_update(scale_grad_2[0], grad_frag[i1, j] * inp)
                            T.reducer_update(base_grad_reducer[j], grad_frag[i1, j])
                            input_mixes_grad[i, j] = grad_frag[i1, j] * mhc_scale[2]

            T.finalize_reducer(scale_grad_0, scale_grad_0_frag)
            T.finalize_reducer(scale_grad_1, scale_grad_1_frag)
            T.finalize_reducer(scale_grad_2, scale_grad_2_frag)
            T.finalize_reducer(base_grad_reducer, base_grad_frag)
            mhc_scale_grad_partial[pid, 0] = scale_grad_0_frag[0]
            mhc_scale_grad_partial[pid, 1] = scale_grad_1_frag[0]
            mhc_scale_grad_partial[pid, 2] = scale_grad_2_frag[0]
            T.copy(base_grad_frag, mhc_base_grad_partial[pid, :])


@tilelang.jit
def mhc_pre_split_mixes_bwd_mhc4_asc(
    pre_layer_mix_grad,
    post_layer_mix_grad,
    comb_res_mix_grad,
    input_mixes,
    post_layer_mix,
    mhc_scale,
    mhc_base,
    input_mixes_grad,
    mhc_scale_grad_partial,
    mhc_base_grad_partial,
    mhc_post_mult_value: float,
    token_block_size: int,
    num_stages: int,
    num_vec_cores: int,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult = 4
    mhc_mult2 = 16
    mhc_mult3 = 24

    pre_layer_mix_grad: T.Tensor[(num_tokens, mhc_mult), T.float32]
    post_layer_mix_grad: T.Tensor[(num_tokens, mhc_mult), T.float32]
    comb_res_mix_grad: T.Tensor[(num_tokens, mhc_mult2), T.float32]
    input_mixes: T.Tensor[(num_tokens, mhc_mult3), T.float32]
    post_layer_mix: T.Tensor[(num_tokens, mhc_mult), T.float32]
    mhc_scale: T.Tensor[(3,), T.float32]
    mhc_base: T.Tensor[mhc_mult3, T.float32]
    input_mixes_grad: T.Tensor[(num_tokens, mhc_mult3), T.float32]
    mhc_scale_grad_partial: T.Tensor[(num_vec_cores, 3), T.float32]
    mhc_base_grad_partial: T.Tensor[(num_vec_cores, mhc_mult3), T.float32]

    with T.Kernel(num_vec_cores) as pid:
        scale_ub = T.alloc_shared((3, 8), T.float32)
        base_ub = T.alloc_shared((mhc_mult3, 8), T.float32)
        input_mixes_ub = T.alloc_shared((token_block_size, mhc_mult3), T.float32)
        input_mixes_grad_ub = T.alloc_shared((token_block_size, mhc_mult3), T.float32)
        pre_layer_mix_grad_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        post_layer_mix_grad_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        post_layer_mix_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        comb_res_mix_grad_ub = T.alloc_shared((token_block_size, mhc_mult2), T.float32)
        # Keep six vector-lane partials across persistent token blocks: three
        # scale reductions plus pre/post/comb base reductions.  A full-mask
        # S.vsts writes 64 fp32 values, so every accumulator row is 256B.
        scale_grad_acc_ub = T.alloc_shared((3, 64), T.float32)
        base_grad_acc_ub = T.alloc_shared((3, 64), T.float32)
        # Scalar reduction results use separate 32B-aligned rows.
        scale_grad_out_ub = T.alloc_shared((3, 8), T.float32)
        base_grad_out_ub = T.alloc_shared((mhc_mult3, 8), T.float32)
        T.annotate_buffer_versions(
            {
                input_mixes_ub: num_stages,
                input_mixes_grad_ub: num_stages,
                pre_layer_mix_grad_ub: num_stages,
                post_layer_mix_grad_ub: num_stages,
                post_layer_mix_ub: num_stages,
                comb_res_mix_grad_ub: num_stages,
            }
        )

        with T.SimtVF(threads=32):
            for i in T.Parallel(3):
                scale_ub[i, 0] = mhc_scale[i]
            for i in T.Parallel(mhc_mult3):
                base_ub[i, 0] = mhc_base[i]

        with T.SimdVF():
            acc_init_mask = S.pset(32)
            acc_init_zeros = S.vdup(T.float32(0), T.float32, acc_init_mask)
            # These UB buffers carry vector-lane partials across persistent
            # token blocks.  Initialize the buffers themselves: initializing
            # temporary registers would leave the first iteration reading stale
            # UB contents.
            for i in T.unroll(3, explicit=True):
                S.vsts(scale_grad_acc_ub[i, 0], acc_init_zeros, acc_init_mask)
                S.vsts(base_grad_acc_ub[i, 0], acc_init_zeros, acc_init_mask)
            S.mem_bar('VST_VLD')

        for t in T.Persistent([T.ceildiv(num_tokens, token_block_size)], num_vec_cores, pid, group_size=1, num_stages=num_stages):
            T.copy(input_mixes[t * token_block_size, 0], input_mixes_ub)
            T.copy(pre_layer_mix_grad[t * token_block_size, 0], pre_layer_mix_grad_ub)
            T.copy(post_layer_mix_grad[t * token_block_size, 0], post_layer_mix_grad_ub)
            T.copy(post_layer_mix[t * token_block_size, 0], post_layer_mix_ub)
            T.copy(comb_res_mix_grad[t * token_block_size, 0], comb_res_mix_grad_ub)

            with T.SimdVF():
                mask = S.pset(32)
                idx = S.vci(T.int32(0), T.int32)
                col3 = S.vand(idx, S.vdup(T.int32(3), T.int32, mask), mask)
                col15 = S.vand(idx, S.vdup(T.int32(15), T.int32, mask), mask)
                zeros = S.vdup(T.float32(0), T.float32, mask)
                ones = S.vdup(T.float32(1), T.float32, mask)
                scale0 = S.vld(scale_ub[0, 0], 'BRC_B32')
                scale1 = S.vld(scale_ub[1, 0], 'BRC_B32')
                post_mult = S.vdup(T.float32(mhc_post_mult_value), T.float32, mask)
                pre_base_vec = S.alloc_var(T.float32)
                post_base_vec = S.alloc_var(T.float32)
                comb_base_vec = S.alloc_var(T.float32)
                pre_base_vec = zeros
                post_base_vec = zeros
                comb_base_vec = zeros
                for base_id in range(4):
                    base_mask = S.vcmps(col3, T.int32(base_id), mask, 'eq')
                    pre_base_vec = S.vsel(S.vld(base_ub[base_id, 0], 'BRC_B32'), pre_base_vec, base_mask)
                    post_base_vec = S.vsel(S.vld(base_ub[base_id + 4, 0], 'BRC_B32'), post_base_vec, base_mask)
                for base_id in range(16):
                    base_mask = S.vcmps(col15, T.int32(base_id), mask, 'eq')
                    comb_base_vec = S.vsel(S.vld(base_ub[base_id + 8, 0], 'BRC_B32'), comb_base_vec, base_mask)

                scale_acc = S.alloc_local((2,), T.float32)
                base_acc = S.alloc_local((2,), T.float32)
                for i in T.unroll(2, explicit=True):
                    scale_acc[i] = S.vld(scale_grad_acc_ub[i, 0])
                    base_acc[i] = S.vld(base_grad_acc_ub[i, 0])

                # The last persistent block may contain fewer than 64 tokens.
                # The hot full-block specialization uses the all-lane mask;
                # only the final block materializes per-vector tail masks.
                valid_tokens = T.min(token_block_size, num_tokens - t * token_block_size)
                for vec_id in T.serial(4):
                    token = S.vadd(S.vshrs(idx, T.int32(2), mask), S.vdup(T.int32(vec_id * 16), T.int32, mask), mask)
                    token4 = S.vshls(token, T.int32(2), mask)
                    token24 = S.vadd(S.vshls(token, T.int32(4), mask), S.vshls(token, T.int32(3), mask), mask)
                    pre_index = S.vadd(token24, col3, mask)
                    post_index = S.vadds(pre_index, T.int32(4), mask)
                    grad_index = S.vadd(token4, col3, mask)
                    token_mask = S.vcmps(token, valid_tokens, mask, 'lt')

                    pre_inp = S.vgather2(input_mixes_ub[0, 0], _u32_index(pre_index), token_mask)
                    pre_grad_out = S.vgather2(pre_layer_mix_grad_ub[0, 0], _u32_index(grad_index), token_mask)
                    pre_act = S.vdiv(
                        ones, S.vadds(S.vexpdif(zeros, S.vadd(S.vmul(pre_inp, scale0, mask), pre_base_vec, mask), mask), T.float32(1), mask), mask
                    )
                    pre_grad = S.vmul(S.vmul(pre_grad_out, pre_act, mask), S.vsub(ones, pre_act, mask), mask)
                    S.vscatter(S.vmul(pre_grad, scale0, mask), input_mixes_grad_ub[0, 0], _u32_index(pre_index), token_mask)
                    scale_acc[0] = S.vadd(scale_acc[0], S.vmul(pre_grad, pre_inp, mask), mask)
                    base_acc[0] = S.vadd(base_acc[0], pre_grad, mask)

                    post_inp = S.vgather2(input_mixes_ub[0, 0], _u32_index(post_index), token_mask)
                    post_grad_out = S.vgather2(post_layer_mix_grad_ub[0, 0], _u32_index(grad_index), token_mask)
                    post_act = S.vgather2(post_layer_mix_ub[0, 0], _u32_index(grad_index), token_mask)
                    post_grad = S.vmul(S.vmul(post_grad_out, post_act, mask), S.vsub(ones, S.vdiv(post_act, post_mult, mask), mask), mask)
                    S.vscatter(S.vmul(post_grad, scale1, mask), input_mixes_grad_ub[0, 0], _u32_index(post_index), token_mask)
                    scale_acc[1] = S.vadd(scale_acc[1], S.vmul(post_grad, post_inp, mask), mask)
                    base_acc[1] = S.vadd(base_acc[1], post_grad, mask)

                for i in T.unroll(2, explicit=True):
                    S.vsts(scale_grad_acc_ub[i, 0], scale_acc[i], mask)
                    S.vsts(base_grad_acc_ub[i, 0], base_acc[i], mask)

            with T.SimdVF():
                mask = S.pset(32)
                idx = S.vci(T.int32(0), T.int32)
                col15 = S.vand(idx, S.vdup(T.int32(15), T.int32, mask), mask)
                scale2 = S.vld(scale_ub[2, 0], 'BRC_B32')
                comb_scale_sum = S.alloc_var(T.float32)
                comb_base_sum = S.alloc_var(T.float32)
                comb_scale_sum = S.vld(scale_grad_acc_ub[2, 0])
                comb_base_sum = S.vld(base_grad_acc_ub[2, 0])
                valid_tokens = T.min(token_block_size, num_tokens - t * token_block_size)
                for vec_id in T.serial(16):
                    token = S.vadd(S.vshrs(idx, T.int32(4), mask), S.vdup(T.int32(vec_id * 4), T.int32, mask), mask)
                    token16 = S.vshls(token, T.int32(4), mask)
                    token24 = S.vadd(token16, S.vshls(token, T.int32(3), mask), mask)
                    input_index = S.vadd(S.vadds(token24, T.int32(8), mask), col15, mask)
                    grad_index = S.vadd(token16, col15, mask)
                    token_mask = S.vcmps(token, valid_tokens, mask, 'lt')
                    comb_inp = S.vgather2(input_mixes_ub[0, 0], _u32_index(input_index), token_mask)
                    comb_grad = S.vgather2(comb_res_mix_grad_ub[0, 0], _u32_index(grad_index), token_mask)
                    S.vscatter(S.vmul(comb_grad, scale2, mask), input_mixes_grad_ub[0, 0], _u32_index(input_index), token_mask)
                    comb_scale_sum = S.vadd(comb_scale_sum, S.vmul(comb_grad, comb_inp, mask), mask)
                    comb_base_sum = S.vadd(comb_base_sum, comb_grad, mask)

                S.vsts(scale_grad_acc_ub[2, 0], comb_scale_sum, mask)
                S.vsts(base_grad_acc_ub[2, 0], comb_base_sum, mask)

            T.copy(input_mixes_grad_ub, input_mixes_grad[t * token_block_size, 0])

        with T.SimdVF():
            mask = S.pset(32)
            lane0 = S.pset(32, 'PAT_VL1')
            idx = S.vci(T.int32(0), T.int32)
            col3 = S.vand(idx, S.vdup(T.int32(3), T.int32, mask), mask)
            col15 = S.vand(idx, S.vdup(T.int32(15), T.int32, mask), mask)
            for i in T.unroll(3, explicit=True):
                S.vsts(scale_grad_out_ub[i, 0], S.vcadd(S.vld(scale_grad_acc_ub[i, 0]), mask), lane0, 'ONEPT_B32')
            pre_base_acc = S.vld(base_grad_acc_ub[0, 0])
            post_base_acc = S.vld(base_grad_acc_ub[1, 0])
            comb_base_acc = S.vld(base_grad_acc_ub[2, 0])
            for base_id in T.unroll(4, explicit=True):
                base_mask = S.vcmps(col3, T.int32(base_id), mask, 'eq')
                S.vsts(base_grad_out_ub[base_id, 0], S.vcadd(pre_base_acc, base_mask), lane0, 'ONEPT_B32')
                S.vsts(base_grad_out_ub[base_id + 4, 0], S.vcadd(post_base_acc, base_mask), lane0, 'ONEPT_B32')
            for base_id in T.unroll(16, unroll_factor=1):
                base_mask = S.vcmps(col15, T.int32(base_id), mask, 'eq')
                S.vsts(base_grad_out_ub[base_id + 8, 0], S.vcadd(comb_base_acc, base_mask), lane0, 'ONEPT_B32')

        with T.SimtVF(threads=32):
            for i in T.Parallel(3):
                mhc_scale_grad_partial[pid, i] = scale_grad_out_ub[i, 0]
            for i in T.Parallel(mhc_mult3):
                mhc_base_grad_partial[pid, i] = base_grad_out_ub[i, 0]

import tilelang
from tilelang import language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def mhc_multilayer_recompute_cuda(
    initial_residual,
    pre_mix_ptrs,
    layer_output_ptrs,
    post_mix_ptrs,
    comb_mix_ptrs,
    layer_input_ptrs,
    residual_ptrs,
    n_thr: int = 64,
    h_blk: int = 512,
    x_dtype: T.dtype = T.bfloat16,
    use_pdl: bool = False,
):
    assert x_dtype in (T.bfloat16, T.float32)
    n = T.dynamic('n')
    mhc = T.const('mhc')
    h = T.const('h')
    L = T.const('L')
    L_post = T.const('L_post')

    initial_residual: T.Tensor[(n, mhc, h), x_dtype]
    pre_mix_ptrs: T.Tensor[(L,), T.ptr]
    layer_output_ptrs: T.Tensor[(L_post,), T.ptr]
    post_mix_ptrs: T.Tensor[(L_post,), T.ptr]
    comb_mix_ptrs: T.Tensor[(L_post,), T.ptr]
    layer_input_ptrs: T.Tensor[(L,), T.ptr]
    residual_ptrs: T.Tensor[(L_post,), T.ptr]

    num_chunks = h // h_blk

    with T.Kernel(n * num_chunks, threads=n_thr) as i_blk:
        if use_pdl:
            T.pdl_sync()
        i_n = i_blk // num_chunks
        i0_h = (i_blk % num_chunks) * h_blk

        res_local = T.alloc_fragment((mhc, h_blk), T.float32)
        new_res_local = T.alloc_fragment((mhc, h_blk), T.float32)
        layer_input_local = T.alloc_fragment(h_blk, T.float32)
        layer_output_local = T.alloc_fragment(h_blk, T.float32)
        pre_mix_local = T.alloc_fragment(mhc, T.float32)
        post_mix_local = T.alloc_fragment(mhc, T.float32)
        comb_mix_local = T.alloc_fragment((mhc, mhc), T.float32)

        layer_output_shared = T.alloc_shared((2, h_blk), x_dtype)
        pre_mix_shared = T.alloc_shared((2, mhc), T.float32)
        post_mix_shared = T.alloc_shared((2, mhc), T.float32)
        comb_mix_shared = T.alloc_shared((2, mhc, mhc), T.float32)

        T.copy(initial_residual[i_n, 0, i0_h], res_local)

        if L_post > 0:
            layer_output_tensor_0 = T.make_tensor(layer_output_ptrs[0], (n, h), x_dtype)
            pre_mix_tensor_0 = T.make_tensor(pre_mix_ptrs[0], (n, mhc), T.float32)
            post_mix_tensor_0 = T.make_tensor(post_mix_ptrs[0], (n, mhc), T.float32)
            comb_mix_tensor_0 = T.make_tensor(comb_mix_ptrs[0], (n, mhc, mhc), T.float32)
            T.async_copy(layer_output_tensor_0[i_n, i0_h], layer_output_shared[0, :])
            T.async_copy(pre_mix_tensor_0[i_n, 0], pre_mix_shared[0, :])
            T.async_copy(post_mix_tensor_0[i_n, 0], post_mix_shared[0, :])
            T.async_copy(comb_mix_tensor_0[i_n, 0, 0], comb_mix_shared[0, :, :])

        for i_layer in T.serial(L_post):
            layer_input_tensor = T.make_tensor(layer_input_ptrs[i_layer], (n, h), x_dtype)
            output_residual_tensor = T.make_tensor(residual_ptrs[i_layer], (n, mhc, h), x_dtype)

            phase = i_layer % 2

            if i_layer + 1 < L_post:
                next_layer_output_tensor = T.make_tensor(layer_output_ptrs[i_layer + 1], (n, h), x_dtype)
                next_pre_mix_tensor = T.make_tensor(pre_mix_ptrs[i_layer + 1], (n, mhc), T.float32)
                next_post_mix_tensor = T.make_tensor(post_mix_ptrs[i_layer + 1], (n, mhc), T.float32)
                next_comb_mix_tensor = T.make_tensor(comb_mix_ptrs[i_layer + 1], (n, mhc, mhc), T.float32)
                T.async_copy(next_layer_output_tensor[i_n, i0_h], layer_output_shared[1 - phase, :])
                T.async_copy(next_pre_mix_tensor[i_n, 0], pre_mix_shared[1 - phase, :])
                T.async_copy(next_post_mix_tensor[i_n, 0], post_mix_shared[1 - phase, :])
                T.async_copy(next_comb_mix_tensor[i_n, 0, 0], comb_mix_shared[1 - phase, :, :])

            if i_layer + 1 < L_post:
                T.ptx_wait_group(4)
            else:
                T.ptx_wait_group(0)

            T.copy(pre_mix_shared[phase, :], pre_mix_local)

            T.clear(layer_input_local)
            for i_mhc in T.serial(mhc):
                for i1_h in T.Parallel(h_blk):
                    layer_input_local[i1_h] += pre_mix_local[i_mhc] * res_local[i_mhc, i1_h]

            T.copy(layer_input_local, layer_input_tensor[i_n, i0_h])

            T.copy(post_mix_shared[phase, :], post_mix_local)
            T.copy(comb_mix_shared[phase, :, :], comb_mix_local)
            T.copy(layer_output_shared[phase, :], layer_output_local)
            for i_mhco, i1_h in T.Parallel(mhc, h_blk):
                new_res_local[i_mhco, i1_h] = T.fmul(post_mix_local[i_mhco], layer_output_local[i1_h])
            for i_mhci in T.serial(mhc):
                for i_mhco, i1_h in T.Parallel(mhc, h_blk):
                    new_res_local[i_mhco, i1_h] = T.fma(comb_mix_local[i_mhci, i_mhco], res_local[i_mhci, i1_h], new_res_local[i_mhco, i1_h])

            T.copy(new_res_local, output_residual_tensor[i_n, 0, i0_h])

            for i_mhc, i1_h in T.Parallel(mhc, h_blk):
                if x_dtype == T.bfloat16:
                    res_local[i_mhc, i1_h] = T.cast(T.cast(new_res_local[i_mhc, i1_h], T.bfloat16), T.float32)
                else:
                    res_local[i_mhc, i1_h] = new_res_local[i_mhc, i1_h]

        if L > L_post:
            pre_mix_tensor_last = T.make_tensor(pre_mix_ptrs[L_post], (n, mhc), T.float32)
            layer_input_tensor_last = T.make_tensor(layer_input_ptrs[L_post], (n, h), x_dtype)

            T.copy(pre_mix_tensor_last[i_n, 0], pre_mix_local)

            T.clear(layer_input_local)
            for i_mhc in T.serial(mhc):
                for i1_h in T.Parallel(h_blk):
                    layer_input_local[i1_h] += pre_mix_local[i_mhc] * res_local[i_mhc, i1_h]

            T.copy(layer_input_local, layer_input_tensor_last[i_n, i0_h])
        if use_pdl:
            T.pdl_trigger()

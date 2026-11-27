import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from tile_kernels.mhc.common_asc import _VEC, _load_f32, _store_f32

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def mhc_multilayer_recompute_asc(
    initial_residual,
    pre_mix_ptrs,
    layer_output_ptrs,
    post_mix_ptrs,
    comb_mix_ptrs,
    layer_input_ptrs,
    residual_ptrs,
    h_blk: int,
    num_vec_cores: int,
    x_dtype: T.dtype = T.bfloat16,
):
    assert x_dtype in (T.bfloat16, T.float32)
    num_tokens = T.dynamic('n')
    mhc = T.const('mhc')
    hidden = T.const('h')
    num_layers = T.const('L')
    num_post_layers = T.const('L_post')

    initial_residual: T.Tensor[(num_tokens, mhc, hidden), x_dtype]
    pre_mix_ptrs: T.Tensor[(num_layers,), T.ptr]
    layer_output_ptrs: T.Tensor[(num_post_layers,), T.ptr]
    post_mix_ptrs: T.Tensor[(num_post_layers,), T.ptr]
    comb_mix_ptrs: T.Tensor[(num_post_layers,), T.ptr]
    layer_input_ptrs: T.Tensor[(num_layers,), T.ptr]
    residual_ptrs: T.Tensor[(num_post_layers,), T.ptr]

    num_hidden_tiles = T.ceildiv(hidden, h_blk)
    num_chunks = h_blk // _VEC
    mhc_pad = T.ceildiv(mhc, 8) * 8  # TODO remove manual padding

    with T.Kernel(num_vec_cores) as core_id:
        residual_ub = T.alloc_shared((2, mhc, h_blk), x_dtype)
        layer_output_ub = T.alloc_shared((2, h_blk), x_dtype)
        layer_input_ub = T.alloc_shared((2, h_blk), x_dtype)
        pre_mix_ub = T.alloc_shared((2, mhc_pad), T.float32)
        post_mix_ub = T.alloc_shared((2, mhc_pad), T.float32)
        comb_mix_ub = T.alloc_shared((2, mhc, mhc), T.float32)

        T.ascend_set_flag('MTE3_MTE2', 0)

        for pid_n, pid_h in T.Persistent([num_tokens, num_hidden_tiles], num_vec_cores, core_id, group_size=1, num_stages=0):
            T.ascend_set_flag('V_MTE2', 0)
            T.ascend_set_flag('V_MTE2', 1)
            T.ascend_set_flag('MTE3_V', 0)
            T.ascend_set_flag('MTE3_V', 1)

            T.ascend_wait_flag('MTE3_MTE2', 0)
            T.copy(initial_residual[pid_n, 0, pid_h * h_blk], residual_ub[0, :, :])
            T.ascend_set_flag('MTE2_V', 2)
            T.ascend_wait_flag('MTE2_V', 2)

            for l in T.serial(num_post_layers):
                input_slot = l % 2
                output_slot = 1 - input_slot
                pre_mix_tensor = T.make_tensor(pre_mix_ptrs[l], (num_tokens, mhc), T.float32)
                post_mix_tensor = T.make_tensor(post_mix_ptrs[l], (num_tokens, mhc), T.float32)
                comb_mix_tensor = T.make_tensor(comb_mix_ptrs[l], (num_tokens, mhc, mhc), T.float32)
                layer_output_tensor = T.make_tensor(layer_output_ptrs[l], (num_tokens, hidden), x_dtype)
                layer_input_tensor = T.make_tensor(layer_input_ptrs[l], (num_tokens, hidden), x_dtype)
                residual_tensor = T.make_tensor(residual_ptrs[l], (num_tokens, mhc, hidden), x_dtype)

                T.ascend_wait_flag('V_MTE2', input_slot)
                T.copy(pre_mix_tensor[pid_n, 0], pre_mix_ub[input_slot, 0:mhc])
                T.copy(post_mix_tensor[pid_n, 0], post_mix_ub[input_slot, 0:mhc])
                T.copy(comb_mix_tensor[pid_n, 0, 0], comb_mix_ub[input_slot, :, :])
                T.copy(layer_output_tensor[pid_n, pid_h * h_blk], layer_output_ub[input_slot, :])
                T.ascend_set_flag('MTE2_V', input_slot)

                T.ascend_wait_flag('MTE2_V', input_slot)
                T.ascend_wait_flag('MTE3_V', output_slot)
                with T.SimdVF():
                    pre_mix = S.alloc_local((mhc,), T.float32)
                    post_mix = S.alloc_local((mhc,), T.float32)
                    comb_mix = S.alloc_local((mhc * mhc,), T.float32)
                    for mi in T.unroll(mhc, explicit=True):
                        pre_mix[mi] = S.vld(pre_mix_ub[input_slot, mi], dist='BRC_B32')
                        post_mix[mi] = S.vld(post_mix_ub[input_slot, mi], dist='BRC_B32')
                    for mi in T.unroll(mhc, explicit=True):
                        for mo in T.unroll(mhc, explicit=True):
                            comb_mix[mi * mhc + mo] = S.vld(comb_mix_ub[input_slot, mi, mo], dist='BRC_B32')
                    residual = S.alloc_local((mhc,), T.float32)
                    output = S.alloc_local((mhc,), T.float32)
                    layer_input = S.alloc_var(T.float32)
                    for chunk_index in T.serial(num_chunks):
                        column_offset = chunk_index * _VEC
                        for mi in T.unroll(mhc, explicit=True):
                            residual[mi] = _load_f32(residual_ub[input_slot, mi, column_offset])
                        layer_input = S.vmul(residual[0], pre_mix[0])
                        for mi in T.unroll(mhc - 1, explicit=True):
                            S.vmula(layer_input, residual[mi + 1], pre_mix[mi + 1])
                        _store_f32(layer_input_ub[input_slot, column_offset], layer_input)
                        layer_output = _load_f32(layer_output_ub[input_slot, column_offset])
                        for mo in T.unroll(mhc, explicit=True):
                            output[mo] = S.vmul(layer_output, post_mix[mo])
                        for mi in T.unroll(mhc, explicit=True):
                            for mo in T.unroll(mhc, explicit=True):
                                S.vmula(output[mo], residual[mi], comb_mix[mi * mhc + mo])
                        for mo in T.unroll(mhc, explicit=True):
                            _store_f32(residual_ub[output_slot, mo, column_offset], output[mo])
                T.ascend_set_flag('V_MTE2', input_slot)
                T.ascend_set_flag('V_MTE3', input_slot)

                T.ascend_wait_flag('V_MTE3', input_slot)
                T.copy(residual_ub[output_slot, :, :], residual_tensor[pid_n, 0, pid_h * h_blk])
                T.copy(layer_input_ub[input_slot, :], layer_input_tensor[pid_n, pid_h * h_blk])
                T.ascend_set_flag('MTE3_V', output_slot)

            T.ascend_set_flag('MTE3_MTE2', 0)

            if num_layers > num_post_layers:
                final_slot = num_post_layers % 2
                final_output_slot = 1 - final_slot
                pre_mix_tensor = T.make_tensor(pre_mix_ptrs[num_post_layers], (num_tokens, mhc), T.float32)
                layer_input_tensor = T.make_tensor(layer_input_ptrs[num_post_layers], (num_tokens, hidden), x_dtype)
                T.ascend_wait_flag('V_MTE2', final_slot)
                T.copy(pre_mix_tensor[pid_n, 0], pre_mix_ub[final_slot, 0:mhc])
                T.ascend_set_flag('MTE2_V', final_slot)
                T.ascend_wait_flag('MTE2_V', final_slot)
                T.ascend_wait_flag('MTE3_V', final_output_slot)
                with T.SimdVF():
                    pre_mix = S.alloc_local((mhc,), T.float32)
                    for mi in T.unroll(mhc, explicit=True):
                        pre_mix[mi] = S.vld(pre_mix_ub[final_slot, mi], dist='BRC_B32')
                    residual = S.alloc_local((mhc,), T.float32)
                    layer_input = S.alloc_var(T.float32)
                    for chunk_index in T.serial(num_chunks):
                        column_offset = chunk_index * _VEC
                        for mi in T.unroll(mhc, explicit=True):
                            residual[mi] = _load_f32(residual_ub[final_slot, mi, column_offset])
                        layer_input = S.vmul(residual[0], pre_mix[0])
                        for mi in T.unroll(mhc - 1, explicit=True):
                            S.vmula(layer_input, residual[mi + 1], pre_mix[mi + 1])
                        _store_f32(layer_input_ub[final_slot, column_offset], layer_input)
                T.ascend_set_flag('V_MTE2', final_slot)
                T.ascend_set_flag('V_MTE3', final_slot)
                T.ascend_wait_flag('V_MTE3', final_slot)
                T.copy(layer_input_ub[final_slot, :], layer_input_tensor[pid_n, pid_h * h_blk])
                T.ascend_set_flag('MTE3_V', final_output_slot)

            T.ascend_wait_flag('V_MTE2', 0)
            T.ascend_wait_flag('V_MTE2', 1)
            T.ascend_wait_flag('MTE3_V', 0)
            T.ascend_wait_flag('MTE3_V', 1)

        T.ascend_wait_flag('MTE3_MTE2', 0)

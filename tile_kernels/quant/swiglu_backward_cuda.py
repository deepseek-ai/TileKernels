from typing import Optional

import tilelang
from tilelang import language as T

from tile_kernels.quant.common import *
from tile_kernels.quant.templates_cuda import PREFETCH_L2_SOURCE, SYMMETRIC_CLAMP_SOURCE, UPPER_CLAMP_SOURCE
from tile_kernels.utils import align


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_swiglu_backward_and_per_token_cast_kernel_cuda(
    hidden: int,
    num_experts: Optional[int],
    alignment: int,
    with_weight: bool,
    with_routed_scaling: bool,
    use_clamp: bool,
    x_dtype: T.dtype,
    act_x_grad_dtype: T.dtype,
    out_config: CastOutputConfig,
    do_recompute: bool = False,
    use_pdl: bool = False,
):
    num_threads = 64
    align_length = 512
    hidden_aligned = align(hidden, align_length)
    use_prefetch = hidden_aligned // align_length > 1
    x_prefetch_elems = 128 // x_dtype.bytes
    grad_prefetch_elems = 128 // act_x_grad_dtype.bytes

    # When quantizing (with_sf), x_grad/out are the unquantized BF16 outputs;
    # otherwise their dtype follows the configured output format (e.g. bf16 or fp32).
    out_dtype = T.bfloat16 if out_config.with_sf else out_config.dtype

    # Runtime symbols
    num_expanded_tokens = T.dynamic('num_expanded_tokens')
    sf_stride = T.dynamic('sf_stride')

    # Input and output share the same num_per_channels when quantized
    if out_config.with_sf:
        _, num_per_channels = out_config.sf_block
        num_sf_per_align = align_length // num_per_channels
        sf_shape = get_sf_shape((num_expanded_tokens, hidden * 2), out_config)
        # Input x_sf always uses standard row-major float32 format.
        in_sf_shape = (num_expanded_tokens, 2 * hidden // num_per_channels)
    else:
        num_per_channels = 0
        num_sf_per_align = 0
        sf_shape = (1, 1)
        in_sf_shape = (1, 1)

    num_blocks = num_expanded_tokens

    @T.prim_func
    def swiglu_backward_and_per_token_cast_kernel_cuda(
        x: T.Tensor[(num_expanded_tokens, hidden * 2), x_dtype],
        x_sf: T.Tensor[in_sf_shape, T.float32],
        act_x_grad: T.Tensor[(num_expanded_tokens, hidden), act_x_grad_dtype],
        topk_weights: T.Tensor[(num_expanded_tokens,), T.float32],
        routed_scaling_factor: T.float32,
        psum_num_tokens_per_expert: T.Tensor[num_experts, T.int32],
        out: T.Tensor[(num_expanded_tokens, hidden), out_dtype],
        x_grad_fp8: T.Tensor[(num_expanded_tokens, hidden * 2), out_config.dtype],
        x_grad_fp8_sf: T.StridedTensor[sf_shape, (sf_stride, 1), out_config.sf_dtype],
        x_grad: T.Tensor[(num_expanded_tokens, hidden * 2), out_dtype],
        weight_grad: T.Tensor[(num_expanded_tokens,), T.float32],
        clamp_value: T.float32,
    ):
        with T.Kernel(num_blocks, threads=num_threads) as (pid_token,):
            if use_prefetch or use_clamp:
                T.import_source(PREFETCH_L2_SOURCE + SYMMETRIC_CLAMP_SOURCE + UPPER_CLAMP_SOURCE)
            # Reduce register usage to 56 per thread.
            T.annotate_min_blocks_per_sm(18)
            if use_pdl:
                T.pdl_sync()
            x_fragment = T.alloc_fragment((align_length,), x_dtype)
            y_fragment = T.alloc_fragment((align_length,), x_dtype)
            act_x_grad_fragment = T.alloc_fragment((align_length,), T.float32 if out_config.with_sf else act_x_grad_dtype)

            x_grad_fragment = T.alloc_fragment((align_length,), T.float32)
            y_grad_fragment = T.alloc_fragment((align_length,), T.float32)
            out_fragment = T.alloc_fragment((align_length,), out_dtype)

            if out_config.with_sf:
                x_sf_fragment = T.alloc_fragment((num_sf_per_align,), T.float32)
                y_sf_fragment = T.alloc_fragment((num_sf_per_align,), T.float32)
                x_grad_fragment_reshaped = T.reshape(x_grad_fragment, (num_sf_per_align, num_per_channels))
                y_grad_fragment_reshaped = T.reshape(y_grad_fragment, (num_sf_per_align, num_per_channels))
                xmax_fragment = T.alloc_fragment((num_sf_per_align,), T.float32)
                ymax_fragment = T.alloc_fragment((num_sf_per_align,), T.float32)

            if with_weight:
                acc = T.alloc_reducer((1,), T.float32, op='sum')
                acc_result = T.alloc_fragment((1,), T.float32)
                T.reducer_init(acc)

            w = T.alloc_var(T.float32, init=1)

            # Padding detection via prefix-sum of tokens-per-expert.
            # Expert e occupies expanded slots [align(psum[e-1], alignment), psum[e]),
            # so pid_token is padding iff there exists e with psum[e-1] <= pid_token < align(psum[e-1], alignment).
            if num_experts:
                expert_id = T.alloc_var(T.int32, init=0)
                prev_boundary = T.alloc_var(T.int32, init=0)
                boundary = T.alloc_var(T.int32, init=psum_num_tokens_per_expert[0])

                while pid_token >= boundary and expert_id < num_experts:
                    prev_boundary = boundary
                    expert_id += 1
                    if expert_id < num_experts:
                        boundary = psum_num_tokens_per_expert[expert_id]

                # pid_token in [prev_boundary, align(prev_boundary, alignment)) is padding.
                if pid_token < align(prev_boundary, alignment):
                    if with_weight:
                        weight_grad[pid_token] = 0.0
                    for k in T.unroll(hidden_aligned // align_length):
                        if out_config.with_sf:
                            for i in T.Parallel(align_length):
                                x_grad_fp8[pid_token, i + k * align_length] = 0.0
                                x_grad_fp8[pid_token, i + hidden + k * align_length] = 0.0
                            for i in T.Parallel(num_sf_per_align):
                                zero = T.cast(0.0, out_config.sf_dtype)
                                store_sf(x_grad_fp8_sf, zero, pid_token, i + k * num_sf_per_align, out_config)
                                store_sf(x_grad_fp8_sf, zero, pid_token, i + hidden // num_per_channels + k * num_sf_per_align, out_config)
                        for i in T.Parallel(align_length):
                            x_grad[pid_token, i + k * align_length] = 0.0
                            x_grad[pid_token, i + hidden + k * align_length] = 0.0
                        if do_recompute:
                            for i in T.Parallel(align_length):
                                out[pid_token, i + k * align_length] = 0.0
                    T.thread_return()

            if with_weight:
                w = topk_weights[pid_token]
                if with_routed_scaling:
                    w = w * routed_scaling_factor

            for k in T.serial(hidden_aligned // align_length):
                for i in T.Parallel(align_length):
                    if i + k * align_length < hidden:
                        x_fragment[i] = x[pid_token, i + k * align_length]
                        y_fragment[i] = x[pid_token, i + hidden + k * align_length]
                        act_x_grad_fragment[i] = act_x_grad[pid_token, i + k * align_length]

                if out_config.with_sf:
                    for i in T.Parallel(num_sf_per_align):
                        if i + k * num_sf_per_align < hidden // num_per_channels:
                            x_sf_fragment[i] = x_sf[pid_token, i + k * num_sf_per_align]
                            y_sf_fragment[i] = x_sf[pid_token, i + hidden // num_per_channels + k * num_sf_per_align]

                if out_config.with_sf:
                    for i in T.Parallel(align_length):
                        if i + k * align_length < hidden:
                            x_grad_fragment[i] = T.cast(x_fragment[i], T.float32) * x_sf_fragment[i // num_per_channels]
                            y_grad_fragment[i] = T.cast(y_fragment[i], T.float32) * y_sf_fragment[i // num_per_channels]

                if use_prefetch:
                    hint_tid = T.get_thread_binding()
                    next_col = (k + 1) * align_length
                    if next_col < hidden:
                        if hint_tid < align_length // x_prefetch_elems and next_col + hint_tid * x_prefetch_elems < hidden:
                            T.call_extern(T.int32, 'tk_prefetch_l2', T.address_of(x[pid_token, next_col + hint_tid * x_prefetch_elems]))
                            T.call_extern(T.int32, 'tk_prefetch_l2', T.address_of(x[pid_token, hidden + next_col + hint_tid * x_prefetch_elems]))
                        if hint_tid < align_length // grad_prefetch_elems and next_col + hint_tid * grad_prefetch_elems < hidden:
                            T.call_extern(
                                T.int32,
                                'tk_prefetch_l2',
                                T.address_of(act_x_grad[pid_token, next_col + hint_tid * grad_prefetch_elems]),
                            )

                for i in T.Parallel(align_length):
                    if i + k * align_length < hidden:
                        x_fp32 = T.alloc_var(T.float32)
                        y_fp32 = T.alloc_var(T.float32)
                        if out_config.with_sf:
                            x_fp32 = x_grad_fragment[i]
                            y_fp32 = y_grad_fragment[i]
                        else:
                            x_fp32 = T.cast(x_fragment[i], T.float32)
                            y_fp32 = T.cast(y_fragment[i], T.float32)

                        is_clamped_x = x_fp32 > clamp_value if use_clamp else False
                        is_clamped_y = (T.abs(y_fp32) > clamp_value) if use_clamp else False

                        if use_clamp:
                            x_fp32 = T.call_extern(T.float32, 'tk_upper_clamp', x_fp32, clamp_value)
                            y_fp32 = T.call_extern(T.float32, 'tk_symmetric_clamp', y_fp32, clamp_value)

                        act_x_grad_fp32 = T.cast(act_x_grad_fragment[i], T.float32)
                        tmp_fp32 = 1 + T.exp(-x_fp32)
                        s_fp32 = 1.0 / tmp_fp32
                        if with_weight or do_recompute:
                            silu_estimate = x_fp32 * s_fp32
                            finite_tmp_fp32 = T.min(tmp_fp32, T.max_value(T.float32))
                            silu_negative_residual = T.ieee_fmaf(silu_estimate, finite_tmp_fp32, -x_fp32)
                            silu = T.ieee_fmaf(-silu_negative_residual, s_fp32, silu_estimate)
                            act_out = silu * y_fp32
                        if with_weight:
                            T.reducer_update(acc[0], act_x_grad_fp32 * act_out)
                        act_x_grad_fp32_ws = act_x_grad_fp32 * w * s_fp32

                        # Write to fragment
                        if do_recompute:
                            out_fragment[i] = act_out * w
                        x_grad_fragment[i] = T.Select(is_clamped_x, 0.0, act_x_grad_fp32_ws * y_fp32 * (1 + x_fp32 * (1 - s_fp32)))
                        y_grad_fragment[i] = T.Select(is_clamped_y, 0.0, act_x_grad_fp32_ws * x_fp32)

                for i in T.Parallel(align_length):
                    if i + k * align_length < hidden:
                        x_grad[pid_token, i + k * align_length] = x_grad_fragment[i]
                        x_grad[pid_token, i + hidden + k * align_length] = y_grad_fragment[i]
                        if do_recompute:
                            out[pid_token, i + k * align_length] = out_fragment[i]

                if out_config.with_sf:
                    T.reduce_absmax(x_grad_fragment_reshaped, xmax_fragment, dim=1)
                    T.reduce_absmax(y_grad_fragment_reshaped, ymax_fragment, dim=1)

                    for i in T.Parallel(num_sf_per_align):
                        if i + k * num_sf_per_align < hidden // num_per_channels:
                            x_sf, x_sf_inv = get_sf_and_inv(xmax_fragment[i], out_config)
                            y_sf, y_sf_inv = get_sf_and_inv(ymax_fragment[i], out_config)
                            store_sf(x_grad_fp8_sf, x_sf, pid_token, i + k * num_sf_per_align, out_config)
                            store_sf(x_grad_fp8_sf, y_sf, pid_token, i + hidden // num_per_channels + k * num_sf_per_align, out_config)
                            xmax_fragment[i] = x_sf_inv
                            ymax_fragment[i] = y_sf_inv

                    x_grad_fp8_fragment = T.alloc_fragment((align_length,), out_config.dtype)
                    y_grad_pt_fragment = T.alloc_fragment((align_length,), out_config.dtype)

                    for i in T.Parallel(align_length):
                        if i + k * align_length < hidden:
                            x_grad_fp8_fragment[i] = x_grad_fragment[i] * xmax_fragment[i // num_per_channels]
                            y_grad_pt_fragment[i] = y_grad_fragment[i] * ymax_fragment[i // num_per_channels]

                    for i in T.Parallel(align_length):
                        if i + k * align_length < hidden:
                            x_grad_fp8[pid_token, i + k * align_length] = x_grad_fp8_fragment[i]
                            x_grad_fp8[pid_token, i + hidden + k * align_length] = y_grad_pt_fragment[i]

            if with_weight:
                T.finalize_reducer(acc, acc_result)
                if with_routed_scaling:
                    weight_grad[pid_token] = acc_result[0] * routed_scaling_factor
                else:
                    weight_grad[pid_token] = acc_result[0]
            if use_pdl:
                T.pdl_trigger()

    return swiglu_backward_and_per_token_cast_kernel_cuda

from math import gcd
from typing import Optional

import tilelang
from tilelang import language as T

from tile_kernels.quant.common import *
from tile_kernels.utils import is_power_of_two


PREFETCH_SOURCE = """
__device__ __forceinline__ int tk_prefetch_l2(const void* addr) {
    asm volatile("prefetch.global.L2 [%0];" : : "l"(__cvta_generic_to_global(addr)) : "memory");
    return 0;
}
"""


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def get_swiglu_forward_and_per_token_cast_kernel_cuda(
    hidden: int,
    num_experts: Optional[int],
    alignment: int,
    with_weight: bool,
    with_routed_scaling: bool,
    clamp_value: Optional[T.float32],
    count_clamp: bool,
    in_dtype: T.dtype,
    out_config: CastOutputConfig,
    num_sms: int,
    use_pdl: bool = False,
):
    num_threads = 1024
    warp_size = 32
    num_warps = num_threads // warp_size
    num_thread_vec = 1
    vec_size = gcd(hidden, get_best_vectorize_size(in_dtype) * warp_size) // warp_size
    vec_size = max(vec_size, 1)

    group_step = gcd(hidden, num_thread_vec * num_threads * vec_size)
    num_groups = num_thread_vec * num_threads * vec_size // group_step
    group_size = num_threads // num_groups
    use_fast_certificate = in_dtype == T.bfloat16 and out_config.dtype == T.float8_e4m3fn and out_config.round_sf
    prefetch_elems = 128 // in_dtype.bytes

    num_iters = T.ceildiv(hidden, group_step)
    if num_iters >= num_sms:
        actual_num_sms = num_sms
        token_step = num_groups * actual_num_sms
        use_inner_iter = True
    else:
        actual_num_sms = num_sms // num_iters * num_iters
        token_step = num_groups * (actual_num_sms // num_iters)
        use_inner_iter = False

    assert hidden % vec_size == 0, f'hidden must be divisible by {vec_size}'
    assert hidden % 64 == 0, 'hidden must be divisible by 64'
    assert num_threads >= 4 * warp_size, '4 warps are used for per group reduction'
    assert num_warps <= warp_size, 'num warps must not exceed the size of a warp'

    # Runtime symbols
    num_expanded_tokens = T.dynamic('num_expanded_tokens')
    sf_stride = T.dynamic('sf_stride')

    # Scaling factor configs
    if out_config.with_sf:
        _, num_per_channels = out_config.sf_block
        assert is_power_of_two(num_per_channels), 'num_per_channels must be power of two'
        num_reduce_rounds = (num_per_channels // vec_size).bit_length() - 1
        assert num_per_channels >= vec_size and num_reduce_rounds <= 5, 'Make sure warp level reduce is enough'
        sf_shape = get_sf_shape((num_expanded_tokens, hidden), out_config)
    else:
        sf_shape = (1, 1)  # A placeholder shape for None value

    @T.prim_func
    def swiglu_forward_and_per_token_cast_kernel_cuda(
        x: T.Tensor[(num_expanded_tokens, hidden * 2), in_dtype],
        out: T.Tensor[(num_expanded_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[sf_shape, (sf_stride, 1), out_config.sf_dtype],
        psum_num_tokens_per_expert: T.Tensor[num_experts, T.int32],
        topk_weights: T.Tensor[(num_expanded_tokens,), T.float32],
        routed_scaling_factor: T.float32,
        clamped_count: T.Tensor[(4,), T.int64],
    ):
        with T.Kernel(num_sms, threads=num_threads) as pid:
            T.import_source(PREFETCH_SOURCE)
            if use_pdl:
                T.pdl_sync()
            if pid >= actual_num_sms:
                T.thread_return()
            tid = T.get_thread_binding()
            warp_id = tid // warp_size
            lane_id = tid % warp_size
            group_id = tid // group_size
            group_offset = tid % group_size
            if use_inner_iter:
                iter_start = 0
                iter_rounds = num_iters
            else:
                iter_start = pid % num_iters
                iter_rounds = 1

            counts_shared = T.alloc_shared((num_warps * 8,), T.int32)
            if num_experts:
                psum_shared = T.alloc_shared((num_experts,), T.int32)

            xl_local = T.alloc_local((num_thread_vec, vec_size), in_dtype)
            xr_local = T.alloc_local((num_thread_vec, vec_size), in_dtype)
            xl_fp32_local = T.alloc_local((num_thread_vec, vec_size), T.float32)
            xr_fp32_local = T.alloc_local((num_thread_vec, vec_size), T.float32)
            x_local = T.alloc_local((num_thread_vec, vec_size), T.float32)
            counts_local = T.alloc_local(3, T.int32)
            total_count = T.alloc_var(T.int64, init=0)
            reduced_counts = T.alloc_local(8, T.int32)
            x_store_local = T.alloc_local((num_thread_vec, vec_size), out_config.dtype)

            if num_experts:
                expert_id = T.alloc_var(T.int32, init=0)
                boundary = T.alloc_var(T.int32)
            token_id = T.alloc_var(T.int32)
            if use_inner_iter:
                token_id = group_id + pid * num_groups
            else:
                token_id = group_id + (pid // num_iters) * num_groups
            topk_weights_var = T.alloc_var(T.float32)

            if num_experts:
                for i in T.Parallel(align(num_experts, num_threads)):
                    if i < num_experts:
                        psum_shared[i] = psum_num_tokens_per_expert[i]
                T.sync_threads()
                boundary = psum_shared[0]
            T.fill(counts_local, 0)

            for k in T.serial(num_expanded_tokens):
                if num_experts:
                    while token_id >= boundary and expert_id < num_experts:
                        expert_id += 1
                        token_id += align(boundary, alignment) - boundary
                        if expert_id < num_experts:
                            boundary = psum_shared[expert_id]

                    if expert_id >= num_experts:
                        T.loop_break()
                else:
                    if token_id >= num_expanded_tokens:
                        T.loop_break()

                T.assume(token_id < num_expanded_tokens and token_id >= 0)

                if with_weight:
                    topk_weights_var = topk_weights[token_id]
                    if with_routed_scaling:
                        topk_weights_var = topk_weights_var * routed_scaling_factor

                if count_clamp:
                    if group_offset == 0 and iter_start == 0:
                        total_count += T.int64(hidden)

                for iter_id in T.serial(iter_rounds):
                    iter = iter_start + iter_id
                    # Batch load num_thread_vec vectors from global memory
                    for vi in T.unroll(num_thread_vec):
                        token_offset = iter * group_step + (vi * group_size + group_offset) * vec_size
                        if token_offset < hidden:
                            for j in T.vectorized(vec_size):
                                xl_local[vi, j] = x[token_id, token_offset + j]
                            for j in T.vectorized(vec_size):
                                xr_local[vi, j] = x[token_id, hidden + token_offset + j]
                        else:
                            for j in T.vectorized(vec_size):
                                xl_local[vi, j] = T.cast(0, in_dtype)
                            for j in T.vectorized(vec_size):
                                xr_local[vi, j] = T.cast(0, in_dtype)
                        for j in T.vectorized(vec_size):
                            xl_fp32_local[vi, j] = T.float32(xl_local[vi, j])
                            xr_fp32_local[vi, j] = T.float32(xr_local[vi, j])

                    # Issue hints at 128-byte intervals for the next token in this expert.
                    next_token = token_id + token_step
                    if next_token >= 0 and next_token < num_expanded_tokens:
                        if next_token < (boundary if num_experts else num_expanded_tokens):
                            for prefetch_vi in T.unroll(num_thread_vec):
                                prefetch_offset = iter * group_step + (prefetch_vi * group_size + group_offset) * vec_size
                                if prefetch_offset < hidden and prefetch_offset % prefetch_elems == 0:
                                    T.call_extern(T.int32, 'tk_prefetch_l2', T.address_of(x[next_token, prefetch_offset]))
                                    T.call_extern(T.int32, 'tk_prefetch_l2', T.address_of(x[next_token, hidden + prefetch_offset]))

                    # Compute swiglu and quantize per thread_vec
                    for vi in T.unroll(num_thread_vec):
                        token_offset = iter * group_step + (vi * group_size + group_offset) * vec_size
                        for j in T.unroll(vec_size):
                            val_l = T.alloc_var(T.float32, init=xl_fp32_local[vi, j])
                            val_r = T.alloc_var(T.float32, init=xr_fp32_local[vi, j])
                            if clamp_value is not None:
                                if count_clamp:
                                    clamp_silu = val_l > clamp_value
                                    val_l = T.Select(clamp_silu, clamp_value, val_l)
                                    counts_local[0] += clamp_silu
                                    clamp_upper = val_r > clamp_value
                                    clamp_lower = val_r < -clamp_value
                                    val_r = T.Select(clamp_upper, clamp_value, val_r)
                                    val_r = T.Select(clamp_lower, -clamp_value, val_r)
                                    counts_local[1] += clamp_upper
                                    counts_local[2] += clamp_lower
                                else:
                                    val_l = T.min(val_l, clamp_value)
                                    val_r = T.max(T.min(val_r, clamp_value), -clamp_value)
                            if use_fast_certificate:
                                val = T.alloc_var(
                                    T.float32,
                                    init=T.call_extern(
                                        T.float32,
                                        '__fdividef',
                                        val_l,
                                        1 + T.call_extern(T.float32, '__expf', -val_l),
                                    )
                                    * val_r,
                                )
                                if with_weight:
                                    val = val * topk_weights_var
                            else:
                                if with_weight:
                                    val = val_l / (1 + T.exp(-val_l)) * val_r * topk_weights_var
                                else:
                                    val = val_l / (1 + T.exp(-val_l)) * val_r
                            x_local[vi, j] = val

                        if use_fast_certificate:
                            # Fast certificate is only enabled for bf16 -> e4m3 with
                            # power-of-two sf. It proves that the fast intrinsic path and
                            # precise path land on the same e4m3/sf decision cell; unsafe
                            # warps fall back to the original precise computation below.
                            cert_e = 32 if with_weight else 16
                            cert_min_metric = T.alloc_var(T.uint32, init=0xFFFFFFFF)
                            cert_max_metric = T.alloc_var(T.uint32, init=0)
                            cert_min_l = T.alloc_local((2,), T.float32)

                            for j in T.vectorized(2):
                                cert_min_l[j] = T.float32(0)
                            # Gate clamp is upper-only, so checking the original gate is
                            # conservative for the fast __expf error domain (gate > -16).
                            for j in T.vectorized(vec_size):
                                cert_min_l[j % 2] = T.min(cert_min_l[j % 2], xl_fp32_local[vi, j])
                            cert_min_l[0] = T.min(cert_min_l[0], cert_min_l[1])
                            # cert_t keeps the low 19 mantissa bits of the fast result
                            # (bits & 0x7ffff). Values too close to either end of this
                            # window are recomputed by the precise fallback.
                            for j in T.unroll(vec_size):
                                cert_t = T.reinterpret(x_local[vi, j], T.uint32) & T.uint32(0x7FFFF)
                                cert_min_metric = T.min(cert_min_metric, cert_t)
                                cert_max_metric = T.max(cert_max_metric, cert_t)
                            if T.all_sync((cert_min_l[0] > -16.0) and (cert_min_metric > cert_e) and (cert_max_metric < 0x80000 - cert_e)) == 0:
                                for j in T.unroll(vec_size):
                                    val_l = T.alloc_var(T.float32, init=xl_fp32_local[vi, j])
                                    val_r = T.alloc_var(T.float32, init=xr_fp32_local[vi, j])
                                    if clamp_value is not None:
                                        # Clamp counts were already accumulated in the main
                                        # path; fallback recomputes values only and must not
                                        # count the same elements again.
                                        val_l = T.min(val_l, clamp_value)
                                        val_r = T.max(T.min(val_r, clamp_value), -clamp_value)
                                    val = T.alloc_var(T.float32, init=(val_l / (1 + T.exp(-val_l))) * val_r)
                                    if with_weight:
                                        val = val * topk_weights_var
                                    x_local[vi, j] = val

                        if out_config.with_sf:
                            # Reduce abs max
                            local_amax = T.alloc_local((2,), T.float32)
                            for j in T.vectorized(2):
                                local_amax[j] = 0
                            for j in T.vectorized(vec_size):
                                local_amax[j % 2] = T.max(local_amax[j % 2], T.abs(x_local[vi, j]))
                            local_amax[0] = T.max(local_amax[0], local_amax[1])
                            for j in T.unroll(num_reduce_rounds):
                                local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 1 << j))

                            # Get scaling factor
                            sf, sf_inv = get_sf_and_inv(local_amax[0], out_config)
                            sf_col = token_offset // num_per_channels
                            if group_offset % (num_per_channels // vec_size) == 0:
                                store_sf(out_sf, sf, token_id, sf_col, out_config)

                            # Apply sf_inv
                            for j in T.vectorized(vec_size):
                                x_local[vi, j] *= sf_inv

                        for j in T.vectorized(vec_size):
                            x_store_local[vi, j] = T.cast(x_local[vi, j], out_config.dtype)

                        if token_offset < hidden:
                            for j in T.vectorized(vec_size):
                                out[token_id, token_offset + j] = x_store_local[vi, j]

                token_id += token_step

            if count_clamp:
                T.sync_threads()
                # Mimic int64 reduction with two int32 reduction
                for i in T.unroll(3):
                    reduced_counts[2 * i] = T.warp_reduce_sum(counts_local[i] & 0xFFFF)
                    reduced_counts[2 * i + 1] = T.warp_reduce_sum(counts_local[i] >> 16)
                reduced_counts[6] = T.warp_reduce_sum(T.int32(total_count & 0xFFFF))
                reduced_counts[7] = T.warp_reduce_sum(T.int32(total_count >> 16))

                if lane_id == 0:
                    for i in T.unroll(8):
                        counts_shared[num_warps * i + warp_id] = reduced_counts[i]

                T.sync_threads()
                if warp_id < 4:
                    count_low = T.alloc_var(T.int32, init=0)
                    count_high = T.alloc_var(T.int32, init=0)
                    if lane_id < num_warps:
                        count_low = counts_shared[num_warps * (2 * warp_id) + lane_id]
                        count_high = counts_shared[num_warps * (2 * warp_id + 1) + lane_id]
                    count_low = T.warp_reduce_sum(count_low)
                    count_high = T.warp_reduce_sum(count_high)

                    if lane_id == 0:
                        T.atomic_add(clamped_count[warp_id], (T.int64(count_high) << 16) + T.int64(count_low))

            if use_pdl:
                T.pdl_trigger()

    return swiglu_forward_and_per_token_cast_kernel_cuda

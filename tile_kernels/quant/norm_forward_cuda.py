import math

import tilelang
from tilelang.cuda import language as T

from tile_kernels.quant.common import CastOutputConfig, get_sf_and_inv, get_sf_index, get_sf_inv_dtype, get_sf_shape
from tile_kernels.utils import align, ceil_div


@tilelang.jit
def get_norm_forward_and_per_token_cast_kernel_cuda(
    hidden: int,
    with_weight: bool,
    with_residual: bool,
    eps: float,
    out_scale: float,
    x_dtype: T.dtype,
    out_config: CastOutputConfig,
    with_out_bf16: bool = False,
    mega_moe_block_m: int = 0,
    use_pdl: bool = False,
):
    assert 0 < hidden <= 8192
    assert out_config.with_sf or out_config.dtype == x_dtype
    assert not out_config.with_sf or out_config.sf_block[0] == 1 and out_config.sf_block[1] in (32, 128)
    assert not with_out_bf16 or out_config.with_sf

    # A row is split into 16-byte vectors, thread `row_offset` of the row holds vectors `vi * row_threads + row_offset`
    num_threads = 128
    num_elems_per_thread = 32
    vec_size = math.gcd(16 // x_dtype.bytes, hidden)
    num_vecs = hidden // vec_size
    num_per_channels = out_config.sf_block[1]
    group_lanes = num_per_channels // vec_size if out_config.with_sf else 1
    row_threads = min(
        num_threads,
        max(
            group_lanes,
            min(8, tilelang.next_power_of_2(num_vecs)),
            tilelang.next_power_of_2(ceil_div(hidden, num_elems_per_thread)),
        ),
    )
    assert not out_config.with_sf or num_per_channels % vec_size == 0 and row_threads % group_lanes == 0

    num_sf_cols = ceil_div(hidden, num_per_channels)
    packed_sf_pad = out_config.use_packed_ue8m0 and out_config.use_tma_aligned_col_major_sf and num_sf_cols % 4 != 0
    sf_inv_dtype = get_sf_inv_dtype(x_dtype, out_config)
    if out_config.with_sf:
        assert hidden % num_per_channels == 0
    num_thread_vec = ceil_div(num_vecs, row_threads)
    block_m = num_threads // row_threads
    warp_threads = min(32, row_threads)
    num_row_warps = row_threads // warp_threads
    row_layout = T.Fragment((block_m, row_threads), forward_thread_fn=lambda i, j: i * row_threads + j)
    warp_layout = T.Fragment(
        (block_m, num_row_warps), forward_thread_fn=lambda i, w, rep: i * row_threads + w * warp_threads + rep, replicate=warp_threads
    )
    warp_reduce_layout = T.Fragment((block_m, num_row_warps, warp_threads), forward_thread_fn=lambda i, w, j: i * row_threads + w * warp_threads + j)
    row_reduce_layout = T.Fragment((block_m, row_threads, num_row_warps), forward_thread_fn=lambda i, j, w: i * row_threads + j)

    mega_sf_tile_rows = 128
    mega_sf_lane_rows = 32
    mega_sf_cols = mega_sf_tile_rows // mega_sf_lane_rows
    if mega_moe_block_m > 0:
        assert out_config.with_sf and num_per_channels == 32 and out_config.round_sf and hidden % mega_sf_tile_rows == 0
    mega_moe_aligned_block_m = align(mega_moe_block_m, mega_sf_tile_rows)

    batch = T.dynamic('batch')
    num_tokens = T.dynamic('num_tokens')
    x_stride = T.dynamic('x_stride')
    x_batch_stride = T.dynamic('x_batch_stride')
    out_stride = T.dynamic('out_stride')
    out_batch_stride = T.dynamic('out_batch_stride')
    out_bf16_stride = T.dynamic('out_bf16_stride')
    out_bf16_batch_stride = T.dynamic('out_bf16_batch_stride')
    residual_stride = T.dynamic('residual_stride')
    residual_batch_stride = T.dynamic('residual_batch_stride')
    residual_out_stride = T.dynamic('residual_out_stride')
    residual_out_batch_stride = T.dynamic('residual_out_batch_stride')
    out_sf_stride = T.dynamic('out_sf_stride')
    out_sf_batch_stride = T.dynamic('out_sf_batch_stride')
    mega_moe_sf_rows = T.dynamic('mega_moe_sf_rows')
    sf_shape = get_sf_shape((num_tokens, hidden), out_config) if out_config.with_sf else (1, 1)

    @T.macro
    def get_mega_moe_sf_row(token_id: int):
        i = token_id % mega_moe_block_m
        return (
            token_id // mega_moe_block_m * mega_moe_aligned_block_m
            + i // mega_sf_tile_rows * mega_sf_tile_rows
            + i % mega_sf_lane_rows * mega_sf_cols
            + i % mega_sf_tile_rows // mega_sf_lane_rows
        )

    @T.prim_func
    def norm_forward_and_per_token_cast_kernel(
        x: T.StridedTensor[(batch, num_tokens, hidden), (x_batch_stride, x_stride, 1), x_dtype],
        out: T.StridedTensor[(batch, num_tokens, hidden), (out_batch_stride, out_stride, 1), out_config.dtype],
        out_sf: T.StridedTensor[(batch, *sf_shape), (out_sf_batch_stride, out_sf_stride, 1), out_config.sf_dtype],
        out_bf16: T.StridedTensor[(batch, num_tokens, hidden), (out_bf16_batch_stride, out_bf16_stride, 1), T.bfloat16],
        mega_moe_sf: T.StridedTensor[(mega_moe_sf_rows, hidden // 128, 4), (4, mega_moe_sf_rows * 4, 1), T.uint8],
        weight: T.Tensor[(hidden,), x_dtype],
        rstd: T.Tensor[(batch, num_tokens), T.float32],
        residual: T.StridedTensor[(batch, num_tokens, hidden), (residual_batch_stride, residual_stride, 1), x_dtype],
        residual_out: T.StridedTensor[(batch, num_tokens, hidden), (residual_out_batch_stride, residual_out_stride, 1), x_dtype],
    ):
        with T.Kernel(T.ceildiv(num_tokens, block_m), batch, threads=num_threads) as (pid_token, batch_id):
            T.assume(x_stride % 16 == 0)
            T.assume(x_batch_stride % 16 == 0)
            T.assume(out_stride % 16 == 0)
            T.assume(out_batch_stride % 16 == 0)
            if with_out_bf16:
                T.assume(out_bf16_stride % 16 == 0)
                T.assume(out_bf16_batch_stride % 16 == 0)
            if with_residual:
                T.assume(residual_stride % 16 == 0)
                T.assume(residual_batch_stride % 16 == 0)
                T.assume(residual_out_stride % 16 == 0)
                T.assume(residual_out_batch_stride % 16 == 0)
            # Keep enough rows in flight for wide rows
            T.annotate_min_blocks_per_sm(5)
            if use_pdl:
                T.pdl_sync()
            tid = T.get_thread_binding()
            row_id = tid // row_threads
            row_offset = tid % row_threads
            token_id = pid_token * block_m + row_id

            sum_partial = T.alloc_fragment((block_m, row_threads), T.float32)
            warp_reducer = T.alloc_reducer((block_m, num_row_warps), T.float32, op='sum')
            warp_fragment = T.alloc_fragment((block_m, num_row_warps), T.float32)
            sum_shared = T.alloc_shared((block_m, num_row_warps), T.float32)
            row_partial = T.alloc_fragment((block_m, row_threads, num_row_warps), T.float32)
            sum_reducer = T.alloc_reducer((block_m, row_threads), T.float32, op='sum')
            sum_fragment = T.alloc_fragment((block_m, row_threads), T.float32)
            T.annotate_layout({sum_partial: row_layout, warp_fragment: warp_layout, row_partial: row_reduce_layout, sum_fragment: row_layout})
            x_local = T.alloc_local((num_thread_vec, vec_size), x_dtype)
            y_fp32_local = T.alloc_local((vec_size,), T.float32)
            y_local = T.alloc_local((num_thread_vec, vec_size), x_dtype)
            sum_local = T.alloc_local((2,), T.float32)
            if with_residual:
                residual_local = T.alloc_local((vec_size,), x_dtype)
            if with_weight:
                weight_local = T.alloc_local((vec_size,), x_dtype)
            if out_config.with_sf:
                sf_local = T.alloc_local((num_thread_vec,), out_config.sf_dtype)
                amax_local = T.alloc_local((num_thread_vec,), T.float32)

            T.clear(sum_local)
            for vi in T.unroll(num_thread_vec):
                token_offset = (vi * row_threads + row_offset) * vec_size
                if token_id < num_tokens and token_offset < hidden:
                    for j in T.vectorized(vec_size):
                        x_local[vi, j] = x[batch_id, token_id, token_offset + j]
                    if with_residual:
                        for j in T.vectorized(vec_size):
                            residual_local[j] = residual[batch_id, token_id, token_offset + j]
                        for j in T.vectorized(vec_size):
                            x_local[vi, j] += residual_local[j]
                        for j in T.vectorized(vec_size):
                            residual_out[batch_id, token_id, token_offset + j] = x_local[vi, j]
                else:
                    T.clear(x_local[vi, :])
                for j in T.vectorized(vec_size):
                    sum_local[j % 2] += T.float32(x_local[vi, j]) * T.float32(x_local[vi, j])

            # Reduce RSTD over the threads of the row
            sum_partial[row_id, row_offset] = sum_local[0] + sum_local[1]
            T.reducer_init(warp_reducer)
            for i, w, j in T.Parallel(block_m, num_row_warps, warp_threads, loop_layout=warp_reduce_layout):
                T.reducer_update(warp_reducer[i, w], sum_partial[i, w * warp_threads + j])
            T.finalize_reducer(warp_reducer, warp_fragment)
            if row_offset % warp_threads == 0:
                sum_shared[row_id, row_offset // warp_threads] = warp_fragment[row_id, row_offset // warp_threads]
            T.sync_threads()
            # Replicate warp sums per thread so the final reducer needs no further CTA barriers.
            for i, j, w in T.Parallel(block_m, row_threads, num_row_warps, loop_layout=row_reduce_layout):
                row_partial[i, j, w] = sum_shared[i, w]
            T.reducer_init(sum_reducer)
            for i, j, w in T.Parallel(block_m, row_threads, num_row_warps, loop_layout=row_reduce_layout):
                T.reducer_update(sum_reducer[i, j], row_partial[i, j, w])
            T.finalize_reducer(sum_reducer, sum_fragment)
            rstd_value = T.alloc_var(T.float32, init=T.rsqrt(sum_fragment[row_id, row_offset] / hidden + eps))
            if row_offset == 0 and token_id < num_tokens:
                rstd[batch_id, token_id] = rstd_value

            # Normalize, rounding to the input dtype
            for vi in T.unroll(num_thread_vec):
                token_offset = (vi * row_threads + row_offset) * vec_size
                is_valid = token_id < num_tokens and token_offset < hidden
                if with_weight:
                    if token_offset < hidden:
                        for j in T.vectorized(vec_size):
                            weight_local[j] = weight[token_offset + j]
                    for j in T.vectorized(vec_size):
                        y_fp32_local[j] = T.float32(x_local[vi, j]) * rstd_value * (T.float32(weight_local[j]) * out_scale)
                else:
                    for j in T.vectorized(vec_size):
                        y_fp32_local[j] = T.float32(x_local[vi, j]) * rstd_value * out_scale
                for j in T.vectorized(vec_size):
                    y_local[vi, j] = y_fp32_local[j]
                if not out_config.with_sf:
                    if is_valid:
                        for j in T.vectorized(vec_size):
                            out[batch_id, token_id, token_offset + j] = y_local[vi, j]
                else:
                    if with_out_bf16 and is_valid:
                        for j in T.vectorized(vec_size):
                            out_bf16[batch_id, token_id, token_offset + j] = y_local[vi, j]

                    # Reduce SF over the threads of the group, as in `per_token_cast`
                    # Rounding is monotonic, so the amax of the rounded output is the rounded FP32 amax
                    amax = T.alloc_var(T.float32, init=0.0)
                    for j in T.unroll(vec_size):
                        amax = T.max(amax, T.abs(y_fp32_local[j]))
                    amax = T.float32(T.cast(amax, x_dtype))
                    for i in T.unroll(int(math.log2(group_lanes))):
                        amax = T.max(amax, T.shfl_xor(amax, 1 << i))
                    amax_local[vi] = amax

            if out_config.with_sf:
                for vi in T.unroll(num_thread_vec):
                    token_offset = (vi * row_threads + row_offset) * vec_size
                    sf, sf_inv = get_sf_and_inv(amax_local[vi], out_config, sf_inv_dtype)
                    sf_local[vi] = sf
                    if token_id < num_tokens and token_offset < hidden:
                        for j in T.vectorized(vec_size):
                            out[batch_id, token_id, token_offset + j] = y_local[vi, j] * sf_inv

            # Store SF after the vector loop, so the vectors are scheduled in a single branch-free region
            if out_config.with_sf:
                if token_id < num_tokens and row_offset % group_lanes == 0:
                    for vi in T.unroll(num_thread_vec):
                        token_offset = (vi * row_threads + row_offset) * vec_size
                        if token_offset < hidden:
                            sf_col = token_offset // num_per_channels
                            sf_row, sf_column = get_sf_index(token_id, sf_col, out_config)
                            out_sf[batch_id, sf_row, sf_column] = sf_local[vi]
                            if packed_sf_pad and sf_col == num_sf_cols - 1:
                                for pad_col in T.unroll(num_sf_cols, align(num_sf_cols, 4)):
                                    _, pad_column = get_sf_index(token_id, pad_col, out_config)
                                    out_sf[batch_id, sf_row, pad_column] = 0
                            if mega_moe_block_m > 0:
                                exponent = (
                                    sf_local[vi] if out_config.use_packed_ue8m0 else T.cast(T.reinterpret(sf_local[vi], T.uint32) >> 23, T.uint8)
                                )
                                mega_moe_sf[get_mega_moe_sf_row(token_id), sf_col // 4, sf_col % 4] = exponent

            if use_pdl:
                T.pdl_trigger()

    return norm_forward_and_per_token_cast_kernel

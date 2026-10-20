import math

import tilelang
# Some TileLang builds (e.g. tilelang-musa) expose only the top-level language module.
try:
    from tilelang.cuda import language as T  # type: ignore
except ImportError:  # pragma: no cover
    from tilelang import language as T  # type: ignore


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }
)
def get_norm_backward_kernel_cuda(
    hidden: int,
    with_weight: bool,
    with_residual_out_grad: bool,
    weight_grad_initialized: bool,
    out_scale: float,
    x_dtype: T.dtype,
    weight_grad_dtype: T.dtype,
    num_sms: int,
):
    warp_size = 32
    # Stage x, out_grad and optional residual_out_grad.
    num_staged_tensors = 3 if with_residual_out_grad else 2
    num_stages = 2

    # Limit cp.async to 16 bytes; divisible wide rows use 256 threads.
    vec_size = max(4 // x_dtype.bytes, math.gcd(16 // x_dtype.bytes, hidden // 32))
    num_threads = 256 if hidden // 128 > 32 and hidden % (256 * vec_size) == 0 else 128
    row_threads = min(num_threads, max(warp_size, tilelang.next_power_of_2(hidden) // 32))
    # Keep padded rows within 64 elements per thread.
    row_threads = max(warp_size, math.gcd(row_threads, hidden // vec_size), tilelang.next_power_of_2(hidden) // 64)
    group_size = num_threads
    num_groups = 1 if group_size == 256 or hidden <= 1024 else 2
    num_threads = num_groups * group_size
    num_rows = num_threads // row_threads
    num_row_warps = row_threads // warp_size
    thread_layout = T.Fragment((num_rows, row_threads), forward_thread_fn=lambda i, j: i * row_threads + j)
    warp_layout = T.Fragment(
        (num_rows, num_row_warps),
        forward_thread_fn=lambda i, w, rep: i * row_threads + w * warp_size + rep,
        replicate=warp_size,
    )
    warp_reduce_layout = T.Fragment((num_rows, num_row_warps, warp_size), forward_thread_fn=lambda i, w, j: i * row_threads + w * warp_size + j)
    row_reduce_layout = T.Fragment((num_rows, row_threads, num_row_warps), forward_thread_fn=lambda i, j, w: i * row_threads + j)
    num_thread_vec = tilelang.cdiv(hidden, row_threads * vec_size)
    block_d = num_thread_vec * row_threads * vec_size

    token_step = num_sms * num_rows

    # Tile the final dW reduction across partial rows and hidden columns.
    block_m = 128
    block_k = 32

    num_tokens = T.dynamic('num_tokens')
    x_stride = T.dynamic('x_stride')
    out_grad_stride = T.dynamic('out_grad_stride')
    residual_out_grad_stride = T.dynamic('residual_out_grad_stride')

    weight_grad_threads = math.gcd(num_threads, block_d)
    weight_grad_vec_size = math.gcd(vec_size, block_d // weight_grad_threads)
    weight_grad_layout = T.Fragment((block_d,), forward_thread_fn=lambda j: (j // weight_grad_vec_size) % weight_grad_threads)

    @T.macro
    def prefetch_vector(src, dst, tensor_id, stage, row_id, token_id, token_offset):
        # Asynchronously load aligned vectors and zero-fill padded rows.
        T.ptx_cp_async(
            T.access_ptr(dst[tensor_id, stage, row_id, token_offset], 'w', vec_size),
            T.access_ptr(src[token_id, token_offset], 'r', vec_size),
            vec_size,
            token_id < num_tokens and token_offset < hidden,
        )

    @T.prim_func
    def norm_backward_kernel(
        x: T.StridedTensor[(num_tokens, hidden), (x_stride, 1), x_dtype],
        weight: T.Tensor[(hidden,), x_dtype],
        out_grad: T.StridedTensor[(num_tokens, hidden), (out_grad_stride, 1), x_dtype],
        rstd: T.Tensor[(num_tokens,), T.float32],
        x_grad: T.Tensor[(num_tokens, hidden), x_dtype],
        weight_grad_partial: T.Tensor[(num_sms, hidden), T.float32],
        weight_grad: T.Tensor[(hidden,), weight_grad_dtype],
        residual_out_grad: T.StridedTensor[(num_tokens, hidden), (residual_out_grad_stride, 1), x_dtype],
    ):
        with T.Kernel(num_sms, threads=num_threads) as pid:
            tid = T.get_thread_binding()
            group_id = tid // group_size
            row_offset = tid % row_threads
            row_id = tid // row_threads

            # Reuse input staging buffers for dW reduction.
            rows_shared = T.alloc_shared((num_staged_tensors, num_stages, num_rows, block_d), x_dtype)
            acc_partial = T.alloc_fragment((num_rows, row_threads), T.float32)
            warp_reducer = T.alloc_reducer((num_rows, num_row_warps), T.float32, op='sum')
            warp_fragment = T.alloc_fragment((num_rows, num_row_warps), T.float32)
            acc_shared = T.alloc_shared((2, num_rows, num_row_warps), T.float32)
            row_partial = T.alloc_fragment((num_rows, row_threads, num_row_warps), T.float32)
            acc_reducer = T.alloc_reducer((num_rows, row_threads), T.float32, op='sum')
            acc_fragment = T.alloc_fragment((num_rows, row_threads), T.float32)
            T.annotate_layout({acc_partial: thread_layout, warp_fragment: warp_layout, row_partial: row_reduce_layout, acc_fragment: thread_layout})
            x_local = T.alloc_local((num_thread_vec, vec_size), T.float32)
            out_grad_local = T.alloc_local((num_thread_vec, vec_size), T.float32)
            if with_residual_out_grad:
                residual_out_grad_local = T.alloc_local((vec_size,), T.float32)
            if with_weight:
                weight_local = T.alloc_local((num_thread_vec, vec_size), T.float32)
                weight_grad_local = T.alloc_local((num_thread_vec, vec_size), T.float32)
                T.clear(weight_grad_local)
                for vi in T.unroll(num_thread_vec):
                    token_offset = (vi * row_threads + row_offset) * vec_size
                    T.copy(weight[token_offset : token_offset + vec_size], weight_local[vi, :])

            # Each thread owns its staged vectors.
            token_start = pid + num_sms * row_id
            for stage in T.unroll(num_stages):
                for vi in T.unroll(num_thread_vec):
                    token_offset = (vi * row_threads + row_offset) * vec_size
                    prefetch_vector(x, rows_shared, 0, stage, row_id, token_start + stage * token_step, token_offset)
            rstd_value = T.alloc_var(T.float32, init=T.if_then_else(token_start < num_tokens, rstd[token_start], 0.0))
            for stage in T.unroll(num_stages):
                for vi in T.unroll(num_thread_vec):
                    token_offset = (vi * row_threads + row_offset) * vec_size
                    prefetch_vector(out_grad, rows_shared, 1, stage, row_id, token_start + stage * token_step, token_offset)
                    if with_residual_out_grad:
                        prefetch_vector(residual_out_grad, rows_shared, 2, stage, row_id, token_start + stage * token_step, token_offset)
                T.ptx_commit_group()

            # Keep loop bounds uniform within each CTA.
            for k in T.serial(T.ceildiv(num_tokens - pid, token_step)):
                token_id = pid + k * token_step + num_sms * row_id
                T.ptx_wait_group(num_stages - 1)
                rstd_next = T.alloc_var(T.float32, init=T.if_then_else(token_id + token_step < num_tokens, rstd[token_id + token_step], 0.0))

                # Load xhat and accumulate the row dot product.
                acc = T.alloc_var(T.float32, init=0.0)
                next_token = token_id + num_stages * token_step
                for vi in T.unroll(num_thread_vec):
                    token_offset = (vi * row_threads + row_offset) * vec_size
                    T.copy(rows_shared[0, k % num_stages, row_id, token_offset : token_offset + vec_size], x_local[vi, :])
                    T.copy(rows_shared[1, k % num_stages, row_id, token_offset : token_offset + vec_size], out_grad_local[vi, :])
                    # Refill after this thread has consumed the vector.
                    prefetch_vector(x, rows_shared, 0, k % num_stages, row_id, next_token, token_offset)
                    prefetch_vector(out_grad, rows_shared, 1, k % num_stages, row_id, next_token, token_offset)
                    for j in T.vectorized(vec_size):
                        x_local[vi, j] *= rstd_value
                        out_grad_local[vi, j] *= out_scale
                        if with_weight:
                            weight_grad_local[vi, j] += x_local[vi, j] * out_grad_local[vi, j]
                            out_grad_local[vi, j] *= weight_local[vi, j]
                    for j in T.unroll(vec_size):
                        acc += x_local[vi, j] * out_grad_local[vi, j]

                if not with_residual_out_grad:
                    T.ptx_commit_group()

                # Reduce within each warp, then publish its sum for the row.
                acc_partial[row_id, row_offset] = acc
                T.reducer_init(warp_reducer)
                for i, w, j in T.Parallel(num_rows, num_row_warps, warp_size, loop_layout=warp_reduce_layout):
                    T.reducer_update(warp_reducer[i, w], acc_partial[i, w * warp_size + j])
                T.finalize_reducer(warp_reducer, warp_fragment)
                # Double-buffer warp sums so the next iteration cannot overwrite active readers.
                if row_offset % warp_size == 0:
                    acc_shared[k % 2, row_id, row_offset // warp_size] = warp_fragment[row_id, row_offset // warp_size]
                T.sync_threads(group_id + 1, group_size)
                # Replicate warp sums per thread so the final reducer needs no further CTA barriers.
                for i, j, w in T.Parallel(num_rows, row_threads, num_row_warps, loop_layout=row_reduce_layout):
                    row_partial[i, j, w] = acc_shared[k % 2, i, w]
                T.reducer_init(acc_reducer)
                for i, j, w in T.Parallel(num_rows, row_threads, num_row_warps, loop_layout=row_reduce_layout):
                    T.reducer_update(acc_reducer[i, j], row_partial[i, j, w])
                T.finalize_reducer(acc_reducer, acc_fragment)

                # Compute and store dX.
                for vi in T.unroll(num_thread_vec):
                    token_offset = (vi * row_threads + row_offset) * vec_size
                    if with_residual_out_grad:
                        T.copy(rows_shared[2, k % num_stages, row_id, token_offset : token_offset + vec_size], residual_out_grad_local)
                        prefetch_vector(residual_out_grad, rows_shared, 2, k % num_stages, row_id, next_token, token_offset)
                    for j in T.vectorized(vec_size):
                        x_grad_value = out_grad_local[vi, j] - x_local[vi, j] * (acc_fragment[row_id, row_offset] / hidden)
                        if with_residual_out_grad:
                            out_grad_local[vi, j] = x_grad_value * rstd_value + residual_out_grad_local[j]
                        else:
                            out_grad_local[vi, j] = x_grad_value * rstd_value
                    if token_id < num_tokens and token_offset < hidden:
                        T.copy(out_grad_local[vi, :], x_grad[token_id, token_offset : token_offset + vec_size])
                if with_residual_out_grad:
                    T.ptx_commit_group()
                rstd_value = rstd_next
            T.ptx_wait_group(0)

            if with_weight:
                # Merge dW into one partial row per CTA using the drained staging buffers.
                T.sync_threads()
                weight_grad_shared = T.view(rows_shared, (num_staged_tensors * num_stages * num_rows * block_d * x_dtype.bytes // 4,), T.float32)
                for vi in T.unroll(num_thread_vec):
                    token_offset = (vi * row_threads + row_offset) * vec_size
                    T.copy(weight_grad_local[vi, :], weight_grad_shared[row_id * block_d + token_offset : row_id * block_d + token_offset + vec_size])
                T.sync_threads()
                weight_grad_fragment = T.alloc_fragment((block_d,), T.float32)
                T.annotate_layout({weight_grad_fragment: weight_grad_layout})
                T.clear(weight_grad_fragment)
                for i in T.unroll(num_rows):
                    for j in T.Parallel(block_d, loop_layout=weight_grad_layout):
                        weight_grad_fragment[j] += weight_grad_shared[i * block_d + j]
                T.copy(weight_grad_fragment, weight_grad_partial[pid, 0])

                T.sync_grid()
                weight_grad_partial_fragment = T.alloc_fragment((block_m, block_k), T.float32)
                weight_grad_acc_fragment = T.alloc_fragment((block_m, block_k), T.float32)
                weight_grad_sum_fragment = T.alloc_fragment((block_k,), T.float32)
                weight_grad_existing_fragment = T.alloc_fragment((block_k,), weight_grad_dtype)
                for pid_hidden in T.serial(pid, T.ceildiv(hidden, block_k), num_sms):
                    T.clear(weight_grad_acc_fragment)
                    for reduce_iter in T.unroll(T.ceildiv(num_sms, block_m)):
                        T.copy(weight_grad_partial[reduce_iter * block_m, pid_hidden * block_k], weight_grad_partial_fragment)
                        for i, j in T.Parallel(block_m, block_k):
                            weight_grad_acc_fragment[i, j] += weight_grad_partial_fragment[i, j]
                    T.reduce_sum(weight_grad_acc_fragment, weight_grad_sum_fragment, dim=0, batch=4)
                    if weight_grad_initialized:
                        T.copy(weight_grad[pid_hidden * block_k : (pid_hidden + 1) * block_k], weight_grad_existing_fragment)
                        for j in T.Parallel(block_k):
                            weight_grad_sum_fragment[j] += T.float32(weight_grad_existing_fragment[j])
                    T.copy(weight_grad_sum_fragment, weight_grad[pid_hidden * block_k : (pid_hidden + 1) * block_k])

    return norm_backward_kernel

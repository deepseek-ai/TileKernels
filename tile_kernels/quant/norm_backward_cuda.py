import math

import tilelang
from tilelang.cuda import language as T

from tile_kernels.config import get_max_smem_per_sm


def _get_norm_backward_schedule(hidden: int, rms_group_size: int, num_sms: int) -> tuple[int, int]:
    """Choose partials and group blocks for a CUDA backward launch."""
    num_rms_groups = hidden // rms_group_size
    block_g = min(
        max(1, tilelang.next_power_of_2(num_rms_groups) // 4),
        max(1, 4096 // tilelang.next_power_of_2(rms_group_size)),
    )
    block_g = math.gcd(num_rms_groups, block_g)
    num_partials = max(1, num_sms // (num_rms_groups // block_g))
    return num_partials, block_g


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }
)
def get_norm_backward_kernel_cuda(
    hidden: int,
    rms_group_size: int,
    with_weight: bool,
    with_residual_out_grad: bool,
    weight_grad_initialized: bool,
    out_scale: float,
    x_dtype: T.dtype,
    out_grad_dtype: T.dtype,
    x_grad_dtype: T.dtype,
    weight_dtype: T.dtype,
    weight_grad_dtype: T.dtype,
    num_sms: int,
    do_recompute: bool,
    with_weight_grad: bool,
):
    num_rms_groups = hidden // rms_group_size
    num_partials, block_g = _get_norm_backward_schedule(hidden, rms_group_size, num_sms)
    # The wrapper chooses whole group blocks, so there is no group tail to mask.
    assert num_rms_groups % block_g == 0
    num_group_blocks = num_rms_groups // block_g
    num_sms = min(num_sms, num_partials * num_group_blocks)
    warp_size = 32
    num_x_staged_tensors = 2 if with_residual_out_grad else 1

    # Pad local rows; global addresses use the unpadded RMS group width.
    access_bytes = max(x_dtype.bytes, out_grad_dtype.bytes, x_grad_dtype.bytes, weight_dtype.bytes)
    vec_size = math.gcd(16 // access_bytes, rms_group_size)
    # Spread each row over lanes so a lane holds few 16-byte vectors: narrow rows give one
    # lane per 16 elements (up to a warp), wide rows up to 128 or 512 lanes.
    num_threads_per_row = max(
        4,
        min(warp_size, 1 << ((rms_group_size // 16).bit_length() - 1)),
        min(128, tilelang.next_power_of_2(rms_group_size) // 32),
        min(512, tilelang.next_power_of_2(rms_group_size) // 64),
    )
    # Add lanes to wide rows when the vector count divides evenly.
    if rms_group_size >= 8192:
        num_threads_per_row = max(num_threads_per_row, math.gcd(512, rms_group_size // vec_size))
    num_thread_vec = tilelang.cdiv(rms_group_size, num_threads_per_row * vec_size)
    row_width = num_thread_vec * num_threads_per_row * vec_size
    num_warps_per_row = max(1, num_threads_per_row // warp_size)
    thread_group_size = max(128, num_threads_per_row)
    bytes_per_row = row_width * (num_x_staged_tensors * x_dtype.bytes + out_grad_dtype.bytes)
    smem_budget = get_max_smem_per_sm() - 2 * 1024
    row_reduce_bytes = 8 * num_warps_per_row if num_warps_per_row > 1 else 0
    target_num_threads = 512 if num_thread_vec * vec_size <= 24 else 256
    num_rows = max(
        1, min(max(target_num_threads, block_g * num_threads_per_row) // num_threads_per_row, smem_budget // (2 * bytes_per_row + row_reduce_bytes))
    )
    num_rows = 1 << (num_rows.bit_length() - 1)
    num_threads = num_rows * num_threads_per_row
    smem_budget -= num_rows * row_reduce_bytes
    staged_row_bytes = num_rows * bytes_per_row
    num_stages = 2 if 2 * staged_row_bytes <= smem_budget else 1
    assert num_stages * staged_row_bytes <= smem_budget
    assert num_threads % warp_size == 0 and num_threads <= 512
    assert num_rows % block_g == 0
    block_m = num_rows // block_g
    token_step = num_partials * block_m
    assert not with_weight_grad or num_stages * num_x_staged_tensors * x_dtype.bytes >= T.float32.bytes

    # Tile the final dW reduction across partial rows and hidden columns.
    weight_grad_block_m = 128
    weight_grad_block_k = 32

    num_tokens = T.dynamic('num_tokens')
    x_stride = T.dynamic('x_stride')
    out_grad_stride = T.dynamic('out_grad_stride')
    residual_out_grad_stride = T.dynamic('residual_out_grad_stride')

    # Pad dW columns for reduction; flatten token rows and columns onto the CTA threads.
    block_d = tilelang.next_power_of_2(block_g * row_width)
    weight_grad_vec_size = max(1, min(4, block_d // num_threads))
    weight_grad_rows_layout = T.Fragment((block_m, block_d), forward_thread_fn=lambda i, j: ((i * block_d + j) // weight_grad_vec_size) % num_threads)

    @T.macro
    def prefetch_vector(src, dst: T.Ref, token_id, rms_group_id, token_offset):
        # Asynchronously load aligned vectors and zero-fill padded rows.
        T.ptx_cp_async(
            T.access_ptr(dst, 'w', vec_size),
            T.access_ptr(src[token_id, rms_group_id * rms_group_size + token_offset], 'r', vec_size),
            vec_size,
            token_id < num_tokens and token_offset < rms_group_size,
        )

    @T.prim_func
    def norm_backward_kernel(
        x: T.StridedTensor[(num_tokens, hidden), (x_stride, 1), x_dtype],
        weight: T.Tensor[(hidden,), weight_dtype],
        out_grad: T.StridedTensor[(num_tokens, hidden), (out_grad_stride, 1), out_grad_dtype],
        rstd: T.Tensor[(num_tokens * num_rms_groups,), T.float32],
        x_grad: T.Tensor[(num_tokens, hidden), x_grad_dtype],
        weight_grad_partial: T.Tensor[(num_partials, hidden), T.float32],
        weight_grad: T.Tensor[(hidden,), weight_grad_dtype],
        residual_out_grad: T.StridedTensor[(num_tokens, hidden), (residual_out_grad_stride, 1), x_dtype],
        out: T.Tensor[(num_tokens, hidden), x_dtype],
    ):
        with T.Kernel(num_sms, threads=num_threads) as pid:
            tid = T.get_thread_binding()
            row_id = tid // num_threads_per_row
            row_offset = tid % num_threads_per_row
            token_row = row_id // block_g

            # Reuse input staging buffers for dW reduction.
            rows_shared = T.alloc_shared((num_stages, num_x_staged_tensors, num_rows, row_width), x_dtype)
            out_grad_shared = T.alloc_shared((num_stages, num_rows, row_width), out_grad_dtype)
            if num_warps_per_row > 1:
                acc_shared = T.alloc_shared((2, num_rows, num_warps_per_row), T.float32)
            x_local = T.alloc_local((num_thread_vec, vec_size), T.float32)
            out_grad_local = T.alloc_local((num_thread_vec, vec_size), T.float32)
            if with_residual_out_grad:
                residual_out_grad_local = T.alloc_local((vec_size,), T.float32)
            for task_id in T.serial(pid, num_partials * num_group_blocks, num_sms):
                partial_id = task_id // num_group_blocks
                group_block = task_id % num_group_blocks
                rms_group_id = group_block * block_g + row_id % block_g
                group_base = rms_group_id * rms_group_size
                if with_weight:
                    weight_local = T.alloc_local((num_thread_vec, vec_size), T.float32)
                    for vi in T.unroll(num_thread_vec):
                        token_offset = (vi * num_threads_per_row + row_offset) * vec_size
                        if token_offset < rms_group_size:
                            T.copy(weight[group_base + token_offset : group_base + token_offset + vec_size], weight_local[vi, :])
                        else:
                            T.clear(weight_local[vi, :])

                if with_weight_grad:
                    weight_grad_local = T.alloc_local((num_thread_vec, vec_size), T.float32)
                    T.clear(weight_grad_local)

                # Each thread owns its staged vectors.
                token_start = partial_id + num_partials * token_row
                for stage in T.unroll(num_stages):
                    for vi in T.unroll(num_thread_vec):
                        token_offset = (vi * num_threads_per_row + row_offset) * vec_size
                        prefetch_vector(x, rows_shared[stage, 0, row_id, token_offset], token_start + stage * token_step, rms_group_id, token_offset)
                rstd_value = T.alloc_var(
                    T.float32,
                    init=T.if_then_else(token_start < num_tokens, rstd[token_start * num_rms_groups + rms_group_id], 0.0),
                )
                for stage in T.unroll(num_stages):
                    for vi in T.unroll(num_thread_vec):
                        token_offset = (vi * num_threads_per_row + row_offset) * vec_size
                        prefetch_vector(
                            out_grad, out_grad_shared[stage, row_id, token_offset], token_start + stage * token_step, rms_group_id, token_offset
                        )
                        if with_residual_out_grad:
                            prefetch_vector(
                                residual_out_grad,
                                rows_shared[stage, 1, row_id, token_offset],
                                token_start + stage * token_step,
                                rms_group_id,
                                token_offset,
                            )
                    T.ptx_commit_group()

                # Keep loop bounds uniform within each CTA.
                buffer_idx = T.alloc_var(T.int32, init=0)
                for token_base in T.serial(partial_id, num_tokens, token_step):
                    token_id = token_base + num_partials * token_row
                    T.ptx_wait_group(num_stages - 1)
                    rstd_next = T.alloc_var(
                        T.float32,
                        init=T.if_then_else(
                            token_id + token_step < num_tokens,
                            rstd[(token_id + token_step) * num_rms_groups + rms_group_id],
                            0.0,
                        ),
                    )

                    # Load xhat and accumulate the row dot product.
                    acc = T.alloc_var(T.float32, init=0.0)
                    next_token = token_id + num_stages * token_step
                    for vi in T.unroll(num_thread_vec):
                        token_offset = (vi * num_threads_per_row + row_offset) * vec_size
                        channel = group_base + token_offset
                        T.copy(rows_shared[buffer_idx % num_stages, 0, row_id, token_offset : token_offset + vec_size], x_local[vi, :])
                        T.copy(out_grad_shared[buffer_idx % num_stages, row_id, token_offset : token_offset + vec_size], out_grad_local[vi, :])
                        # Refill after this thread has consumed the vector.
                        prefetch_vector(x, rows_shared[buffer_idx % num_stages, 0, row_id, token_offset], next_token, rms_group_id, token_offset)
                        prefetch_vector(
                            out_grad, out_grad_shared[buffer_idx % num_stages, row_id, token_offset], next_token, rms_group_id, token_offset
                        )
                        for j in T.vectorized(vec_size):
                            x_local[vi, j] *= rstd_value
                            out_grad_local[vi, j] *= out_scale
                            if with_weight_grad:
                                weight_grad_local[vi, j] += x_local[vi, j] * out_grad_local[vi, j]
                            if with_weight:
                                out_grad_local[vi, j] *= weight_local[vi, j]
                        if do_recompute and token_id < num_tokens and token_offset < rms_group_size:
                            for j in T.vectorized(vec_size):
                                gamma = weight_local[vi, j] if with_weight else T.float32(1.0)
                                out[token_id, channel + j] = x_local[vi, j] * (gamma * out_scale)
                        for j in T.unroll(vec_size):
                            acc += x_local[vi, j] * out_grad_local[vi, j]

                    # Commit before row reduction once every tensor has been prefetched.
                    if not with_residual_out_grad:
                        T.ptx_commit_group()

                    for i in T.unroll(int(math.log2(min(num_threads_per_row, warp_size)))):
                        acc += T.shfl_xor(acc, 1 << i)
                    row_acc = T.alloc_var(T.float32, init=acc)
                    if num_warps_per_row > 1:
                        # Double-buffer warp sums so the next iteration cannot overwrite active readers.
                        if row_offset % warp_size == 0:
                            acc_shared[buffer_idx, row_id, row_offset // warp_size] = acc
                        T.sync_threads(tid // thread_group_size + 1, thread_group_size)
                        row_acc = 0.0
                        for w in T.unroll(num_warps_per_row):
                            row_acc += acc_shared[buffer_idx, row_id, w]

                    # Compute and store dX.
                    for vi in T.unroll(num_thread_vec):
                        token_offset = (vi * num_threads_per_row + row_offset) * vec_size
                        channel = group_base + token_offset
                        if with_residual_out_grad:
                            T.copy(rows_shared[buffer_idx % num_stages, 1, row_id, token_offset : token_offset + vec_size], residual_out_grad_local)
                            prefetch_vector(
                                residual_out_grad,
                                rows_shared[buffer_idx % num_stages, 1, row_id, token_offset],
                                next_token,
                                rms_group_id,
                                token_offset,
                            )
                        for j in T.vectorized(vec_size):
                            x_grad_value = out_grad_local[vi, j] - x_local[vi, j] * (row_acc / rms_group_size)
                            out_grad_local[vi, j] = x_grad_value * rstd_value
                            if with_residual_out_grad:
                                out_grad_local[vi, j] += residual_out_grad_local[j]
                        if token_id < num_tokens and token_offset < rms_group_size:
                            T.copy(out_grad_local[vi, :], x_grad[token_id, channel : channel + vec_size])
                    if with_residual_out_grad:
                        T.ptx_commit_group()
                    rstd_value = rstd_next
                    buffer_idx = buffer_idx ^ 1
                T.ptx_wait_group(0)

                if with_weight_grad:
                    # Merge dW across rows using the drained staging buffers.
                    T.sync_threads()
                    weight_grad_shared = T.view(rows_shared, (num_stages * num_x_staged_tensors * x_dtype.bytes // 4, num_rows, row_width), T.float32)
                    for vi in T.unroll(num_thread_vec):
                        token_offset = (vi * num_threads_per_row + row_offset) * vec_size
                        T.copy(weight_grad_local[vi, :], weight_grad_shared[0, row_id, token_offset : token_offset + vec_size])
                    T.sync_threads()
                    weight_grad_rows_fragment = T.alloc_fragment((block_m, block_d), T.float32)
                    weight_grad_fragment = T.alloc_fragment((block_d,), T.float32)
                    T.annotate_layout({weight_grad_rows_fragment: weight_grad_rows_layout})
                    for i, j in T.Parallel(block_m, block_d):
                        weight_grad_rows_fragment[i, j] = T.if_then_else(
                            j < block_g * row_width, weight_grad_shared[0, i * block_g + j // row_width, j % row_width], 0.0
                        )
                    T.reduce_sum(weight_grad_rows_fragment, weight_grad_fragment, dim=0)
                    for j in T.Parallel(block_d):
                        channel = (group_block * block_g + j // row_width) * rms_group_size + j % row_width
                        if j < block_g * row_width and j % row_width < rms_group_size:
                            weight_grad_partial[partial_id, channel] = weight_grad_fragment[j]
                if with_weight_grad or num_warps_per_row > 1:
                    T.sync_threads()

            if with_weight_grad:
                T.sync_grid()
                weight_grad_partial_fragment = T.alloc_fragment((weight_grad_block_m, weight_grad_block_k), T.float32)
                weight_grad_acc_fragment = T.alloc_fragment((weight_grad_block_m, weight_grad_block_k), T.float32)
                weight_grad_sum_fragment = T.alloc_fragment((weight_grad_block_k,), T.float32)
                weight_grad_existing_fragment = T.alloc_fragment((weight_grad_block_k,), weight_grad_dtype)
                for pid_hidden in T.serial(pid, T.ceildiv(hidden, weight_grad_block_k), num_sms):
                    T.clear(weight_grad_acc_fragment)
                    for reduce_iter in T.unroll(T.ceildiv(num_partials, weight_grad_block_m)):
                        T.copy(weight_grad_partial[reduce_iter * weight_grad_block_m, pid_hidden * weight_grad_block_k], weight_grad_partial_fragment)
                        for i, j in T.Parallel(weight_grad_block_m, weight_grad_block_k):
                            weight_grad_acc_fragment[i, j] += weight_grad_partial_fragment[i, j]
                    T.reduce_sum(weight_grad_acc_fragment, weight_grad_sum_fragment, dim=0, batch=4)
                    if weight_grad_initialized:
                        T.copy(weight_grad[pid_hidden * weight_grad_block_k : (pid_hidden + 1) * weight_grad_block_k], weight_grad_existing_fragment)
                        for j in T.Parallel(weight_grad_block_k):
                            weight_grad_sum_fragment[j] += T.float32(weight_grad_existing_fragment[j])
                    T.copy(weight_grad_sum_fragment, weight_grad[pid_hidden * weight_grad_block_k : (pid_hidden + 1) * weight_grad_block_k])

    return norm_backward_kernel

import tilelang
from tilelang import language as T

from tile_kernels.utils import ceil_div


def _choose_sinkhorn_config(head_dim: int) -> tuple[int, int]:
    for threads in (128, 64, 32):
        if head_dim % (threads * 8) == 0:
            return threads, 8
    if head_dim % 128 == 0:
        return 64, 2
    raise ValueError(f'No valid thread/vector config for head_dim={head_dim}')


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_engram_sinkhorn_momentum_kernel_cuda(
    grad_dtype: T.dtype,
    momentum_dtype: T.dtype,
    nesterov: bool,
    beta1: float,
    alignment: int,
    use_pdl: bool = False,
):
    num_threads = 32
    assert alignment in (128, 256)
    blk_m = alignment
    num_rows = T.dynamic('num_rows')

    @T.prim_func
    def engram_sinkhorn_momentum_kernel_cuda(
        grad: T.Tensor[(num_rows, blk_m), grad_dtype],
        momentum_buffer: T.Tensor[(num_rows, blk_m), momentum_dtype],
        sinkhorn_input: T.Tensor[(num_rows, blk_m), T.float32],
    ):
        with T.Kernel(num_rows, threads=num_threads) as pid:
            if use_pdl:
                T.pdl_sync()
            grad_frag = T.alloc_fragment((blk_m,), T.float)
            momentum_frag = T.alloc_fragment((blk_m,), T.float)
            sinkhorn_frag = T.alloc_fragment((blk_m,), T.float)

            T.copy(grad[pid, :], grad_frag, disable_tma=True)
            T.copy(momentum_buffer[pid, :], momentum_frag, disable_tma=True)

            for i in T.Parallel(blk_m):
                momentum_frag[i] = T.ieee_fmaf(momentum_frag[i], beta1, (1.0 - beta1) * grad_frag[i], 'rn')
                if nesterov:
                    sinkhorn_frag[i] = T.ieee_fmaf(grad_frag[i], 1.0 - beta1, beta1 * momentum_frag[i], 'rn')
                else:
                    sinkhorn_frag[i] = momentum_frag[i]

            T.copy(momentum_frag, momentum_buffer[pid, :], disable_tma=True)
            T.copy(sinkhorn_frag, sinkhorn_input[pid, :], disable_tma=True)
            if use_pdl:
                T.pdl_trigger()

    return engram_sinkhorn_momentum_kernel_cuda


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def get_engram_sinkhorn_step_kernel_cuda(
    head_dim: int,
    eps: float,
    num_ctas: int,
    use_pdl: bool = False,
):
    """Normalize Sinkhorn rows and write deterministic per-CTA column partials."""
    num_rows = T.dynamic('num_rows')
    threads, vec_size = _choose_sinkhorn_config(head_dim)

    def fragment_layout(col):
        thread_id = (col // vec_size) % threads
        local_id = (col // (threads * vec_size)) * vec_size + col % vec_size
        return thread_id, local_id

    @T.prim_func
    def engram_sinkhorn_step_kernel_cuda(
        matrix: T.Tensor[(num_rows, head_dim), T.float32],
        row_scale: T.Tensor[(num_rows,), T.float32],
        col_scale: T.Tensor[(head_dim,), T.float32],
        row_norm: T.Tensor[(num_rows,), T.float32],
        col_sumsq_partial: T.Tensor[(num_ctas, head_dim), T.float32],
    ):
        with T.Kernel(num_ctas, threads=threads) as pid:
            if use_pdl:
                T.pdl_sync()
            thread_idx = T.get_thread_binding()
            row_scale_var = T.alloc_var(T.float32)
            norm_var = T.alloc_var(T.float32)
            norm_frag = T.alloc_fragment((1,), T.float32)
            col_scale_frag = T.alloc_fragment((head_dim,), T.float32)
            col_acc_frag = T.alloc_fragment((head_dim,), T.float32)
            x_frag = T.alloc_fragment((head_dim,), T.float32)
            x_sq_frag = T.alloc_fragment((head_dim,), T.float32)
            T.annotate_layout(
                {
                    col_scale_frag: T.Fragment((head_dim,), forward_fn=fragment_layout),
                    col_acc_frag: T.Fragment((head_dim,), forward_fn=fragment_layout),
                    x_frag: T.Fragment((head_dim,), forward_fn=fragment_layout),
                    x_sq_frag: T.Fragment((head_dim,), forward_fn=fragment_layout),
                }
            )

            T.clear(col_acc_frag)
            T.copy(col_scale, col_scale_frag, disable_tma=True)

            rows_per_cta = T.ceildiv(num_rows, num_ctas)
            for row in T.serial(
                T.min(rows_per_cta * pid, num_rows),
                T.min(rows_per_cta * (pid + 1), num_rows),
            ):
                row_scale_var = row_scale[row]
                for col in T.Parallel(head_dim):
                    x_frag[col] = row_scale_var * matrix[row, col] * col_scale_frag[col]
                    x_sq_frag[col] = x_frag[col] * x_frag[col]
                T.reduce_sum(x_sq_frag, norm_frag)

                norm_var = T.sqrt(norm_frag[0])
                if thread_idx == 0:
                    row_norm[row] = norm_var
                norm_var = 1.0 / (norm_var + eps)
                if thread_idx == 0:
                    row_scale[row] = row_scale_var * norm_var

                for col in T.Parallel(head_dim):
                    x_frag[col] *= norm_var
                    col_acc_frag[col] += x_frag[col] * x_frag[col]

            T.copy(col_acc_frag, col_sumsq_partial[pid, :], disable_tma=True)
            if use_pdl:
                T.pdl_trigger()

    return engram_sinkhorn_step_kernel_cuda


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def get_engram_sinkhorn_step_reduce_kernel_cuda(
    head_dim: int,
    num_ctas: int,
    use_pdl: bool = False,
):
    blk_d = 256 if head_dim % 256 == 0 else 128
    threads = blk_d // 4
    assert head_dim % blk_d == 0
    num_col_blocks = head_dim // blk_d
    num_batches = 8
    num_rows = ceil_div(num_ctas, num_batches)

    @T.prim_func
    def engram_sinkhorn_step_reduce_kernel_cuda(
        col_sumsq_partial: T.Tensor[(num_ctas, head_dim), T.float32],
        col_sumsq: T.Tensor[(head_dim,), T.float32],
    ):
        with T.Kernel(num_col_blocks, threads=threads) as pid:
            if use_pdl:
                T.pdl_sync()
            col_sumsq_shared = T.alloc_shared((num_rows, blk_d), T.float32)
            col_sumsq_frag = T.alloc_fragment((blk_d,), T.float32)
            T.clear(col_sumsq_frag)

            for i_r in T.Pipelined(0, num_batches, num_stages=2):
                T.copy(
                    col_sumsq_partial[
                        i_r * num_rows : (i_r + 1) * num_rows,
                        pid * blk_d : (pid + 1) * blk_d,
                    ],
                    col_sumsq_shared,
                    disable_tma=True,
                )

                for i in T.Serial(num_rows):
                    for j in T.Parallel(blk_d):
                        col_sumsq_frag[j] += col_sumsq_shared[i, j]

            T.copy(col_sumsq_frag, col_sumsq[pid * blk_d : (pid + 1) * blk_d], disable_tma=True)
            if use_pdl:
                T.pdl_trigger()

    return engram_sinkhorn_step_reduce_kernel_cuda


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_engram_sinkhorn_finalize_kernel_cuda(
    head_dim: int,
    eps: float,
    alpha: float,
    use_pdl: bool = False,
):
    """Materialize the final row-normalized Sinkhorn matrix scaled by alpha."""
    num_rows = T.dynamic('num_rows')
    threads, vec_size = _choose_sinkhorn_config(head_dim)

    def fragment_layout(col):
        thread_id = (col // vec_size) % threads
        local_id = (col // (threads * vec_size)) * vec_size + col % vec_size
        return thread_id, local_id

    @T.prim_func
    def engram_sinkhorn_finalize_kernel_cuda(
        matrix: T.Tensor[(num_rows, head_dim), T.float32],
        row_scale: T.Tensor[(num_rows,), T.float32],
        col_scale: T.Tensor[(head_dim,), T.float32],
        out: T.Tensor[(num_rows, head_dim), T.float32],
    ):
        with T.Kernel(num_rows, threads=threads) as pid_r:
            if use_pdl:
                T.pdl_sync()
            row_scale_var = T.alloc_var(T.float32)
            norm_var = T.alloc_var(T.float32)
            norm_frag = T.alloc_fragment((1,), T.float32)
            x_frag = T.alloc_fragment((head_dim,), T.float32)
            x_sq_frag = T.alloc_fragment((head_dim,), T.float32)
            T.annotate_layout(
                {
                    x_frag: T.Fragment((head_dim,), forward_fn=fragment_layout),
                    x_sq_frag: T.Fragment((head_dim,), forward_fn=fragment_layout),
                }
            )

            row_scale_var = row_scale[pid_r]
            for col in T.Parallel(head_dim):
                x_frag[col] = row_scale_var * matrix[pid_r, col] * col_scale[col]
                x_sq_frag[col] = x_frag[col] * x_frag[col]
            T.reduce_sum(x_sq_frag, norm_frag)

            norm_var = alpha / (T.sqrt(norm_frag[0]) + eps)

            for col in T.Parallel(head_dim):
                out[pid_r, col] = x_frag[col] * norm_var
            if use_pdl:
                T.pdl_trigger()

    return engram_sinkhorn_finalize_kernel_cuda

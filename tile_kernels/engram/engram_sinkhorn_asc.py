import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S


@tilelang.jit
def get_engram_sinkhorn_momentum_kernel_asc(
    grad_dtype: T.dtype,
    momentum_dtype: T.dtype,
    nesterov: bool,
    beta1: float,
    alignment: int,
    num_vec_cores: int,
):
    vec_size = 64
    assert alignment in (128, 256)
    rows_per_tile = 4096 // alignment
    num_stages = 4

    num_rows = T.dynamic('num_rows')
    num_chunks = alignment // vec_size
    grad_is_bf16 = grad_dtype == T.bfloat16
    momentum_is_bf16 = momentum_dtype == T.bfloat16

    @T.prim_func
    def engram_sinkhorn_momentum_kernel_asc(
        grad: T.Tensor[(num_rows, alignment), grad_dtype],
        momentum_buffer: T.Tensor[(num_rows, alignment), momentum_dtype],
        sinkhorn_input: T.Tensor[(num_rows, alignment), T.float32],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            grad_ub = T.alloc_shared((rows_per_tile, alignment), grad_dtype)
            momentum_ub = T.alloc_shared((rows_per_tile, alignment), momentum_dtype)
            sinkhorn_ub = T.alloc_shared((rows_per_tile, alignment), T.float32)
            T.annotate_buffer_versions(
                {
                    grad_ub: num_stages,
                    momentum_ub: num_stages,
                    sinkhorn_ub: num_stages,
                }
            )

            for tile_id in T.Persistent(
                [T.ceildiv(num_rows, rows_per_tile)],
                num_vec_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
            ):
                row_start = tile_id * rows_per_tile
                T.copy(grad[row_start : row_start + rows_per_tile, :], grad_ub)
                T.copy(momentum_buffer[row_start : row_start + rows_per_tile, :], momentum_ub)

                with T.SimdVF():
                    beta_vec = S.vdup(beta1, T.float32)
                    scaled_grad = S.alloc_var(T.float32)
                    momentum_f32 = S.alloc_var(T.float32)
                    for row in T.serial(rows_per_tile):
                        for chunk in T.unroll(num_chunks, explicit=True):
                            col = chunk * vec_size
                            if grad_is_bf16:
                                grad_f32 = S.vcvt(S.vld(grad_ub[row, col], dist='UNPK_B16'), T.float32, part=0)
                            else:
                                grad_f32 = S.vld(grad_ub[row, col])
                            if momentum_is_bf16:
                                momentum_f32 = S.vcvt(S.vld(momentum_ub[row, col], dist='UNPK_B16'), T.float32, part=0)
                            else:
                                momentum_f32 = S.vld(momentum_ub[row, col])

                            scaled_grad = S.vmuls(grad_f32, 1.0 - beta1)
                            S.vmadd(momentum_f32, beta_vec, scaled_grad)
                            if momentum_is_bf16:
                                S.vsts(momentum_ub[row, col], S.vcvt(momentum_f32, T.bfloat16), dist='PK_B32')
                            else:
                                S.vsts(momentum_ub[row, col], momentum_f32)

                            if nesterov:
                                S.vaxpy(scaled_grad, momentum_f32, beta1)
                                S.vsts(sinkhorn_ub[row, col], scaled_grad)
                            else:
                                S.vsts(sinkhorn_ub[row, col], momentum_f32)

                T.copy(momentum_ub, momentum_buffer[row_start : row_start + rows_per_tile, :])
                T.copy(sinkhorn_ub, sinkhorn_input[row_start : row_start + rows_per_tile, :])

    return engram_sinkhorn_momentum_kernel_asc


@tilelang.jit
def get_engram_sinkhorn_step_kernel_asc(
    head_dim: int,
    eps: float,
    num_vec_cores: int,
):
    vec_size = 64
    num_accs = 4 if head_dim % 256 == 0 else 2
    assert head_dim % (vec_size * num_accs) == 0
    num_rows = T.dynamic('num_rows')
    num_chunks = head_dim // vec_size
    num_stages = 4
    batch_size = 256

    @T.prim_func
    def engram_sinkhorn_step_kernel_asc(
        matrix: T.Tensor[(num_rows, head_dim), T.float32],
        row_scale: T.Tensor[(num_rows,), T.float32],
        col_scale: T.Tensor[(head_dim,), T.float32],
        row_norm: T.Tensor[(num_rows,), T.float32],
        col_sumsq_partial: T.Tensor[(num_vec_cores, head_dim), T.float32],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            col_scale_ub = T.alloc_shared((head_dim,), T.float32)
            col_sumsq_ub = T.alloc_shared((head_dim,), T.float32)
            matrix_ub = T.alloc_shared((head_dim,), T.float32)
            row_stats_batch_ub = T.alloc_shared((2, batch_size), T.float32)
            T.annotate_buffer_versions(
                {
                    matrix_ub: num_stages,
                    row_stats_batch_ub: num_stages,
                }
            )

            T.copy(col_scale, col_scale_ub)
            with T.SimdVF():
                T.clear(col_sumsq_ub)

            num_batches = T.ceildiv(num_rows, batch_size)
            for batch_id in T.Persistent([num_batches], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
                row_start = batch_id * batch_size
                batch_rows = T.min(batch_size, num_rows - row_start)
                T.copy(row_scale[row_start : row_start + batch_rows], row_stats_batch_ub[0, :batch_rows])
                for output_slot in T.serial(batch_rows):
                    row = row_start + output_slot
                    T.copy(matrix[row, :], matrix_ub)

                    with T.SimdVF():
                        zero = S.vdup(0.0, T.float32)
                        one = S.vdup(1.0, T.float32)
                        accum = S.alloc_local((num_accs,), T.float32)
                        for acc_id in T.unroll(num_accs, explicit=True):
                            accum[acc_id] = zero

                        scale = S.vld(row_stats_batch_ub[0, output_slot], dist='BRC_B32')
                        for chunk_group in T.serial(num_chunks // num_accs):
                            for acc_id in T.unroll(num_accs, explicit=True):
                                chunk = chunk_group * num_accs + acc_id
                                offset = chunk * vec_size
                                base = S.vmul(S.vmul(scale, S.vld(matrix_ub[offset])), S.vld(col_scale_ub[offset]))
                                S.vmula(accum[acc_id], base, base)
                                S.vsts(matrix_ub[offset], base)

                        if num_accs == 4:
                            for acc_id in T.unroll(2, explicit=True):
                                accum[acc_id] = S.vadd(accum[acc_id], accum[acc_id + 2])
                        accum[0] = S.vadd(accum[0], accum[1])

                        norm = S.vsqrt(S.vcadd(accum[0]))
                        inv_norm = S.vdiv(one, S.vadds(norm, eps))
                        normalized_scale = S.vmul(scale, inv_norm)
                        inv_norm_vec = S.vdupv(inv_norm)
                        S.vsts(row_stats_batch_ub[1, output_slot], norm, dist='ONEPT_B32')
                        S.vsts(row_stats_batch_ub[0, output_slot], normalized_scale, dist='ONEPT_B32')

                        S.mem_bar('VST_VLD')
                        col_acc = S.alloc_var(T.float32)
                        normalized_base = S.alloc_var(T.float32)
                        for chunk in T.serial(num_chunks):
                            offset = chunk * vec_size
                            col_acc = S.vld(col_sumsq_ub[offset])
                            normalized_base = S.vmul(S.vld(matrix_ub[offset]), inv_norm_vec)
                            S.vmula(col_acc, normalized_base, normalized_base)
                            S.vsts(col_sumsq_ub[offset], col_acc)

                T.copy(row_stats_batch_ub[1, :batch_rows], row_norm[row_start : row_start + batch_rows])
                T.copy(row_stats_batch_ub[0, :batch_rows], row_scale[row_start : row_start + batch_rows])

            T.ascend_pipe_barrier('PIPE_ALL')
            T.copy(col_sumsq_ub, col_sumsq_partial[core_id, :])

    return engram_sinkhorn_step_kernel_asc


@tilelang.jit
def get_engram_sinkhorn_step_reduce_kernel_asc(
    head_dim: int,
    num_partials: int,
    num_vec_cores: int,
):
    vec_size = 64
    num_accs = 4 if head_dim % 256 == 0 else 2
    blk_d = vec_size * num_accs
    assert head_dim % blk_d == 0
    num_col_blocks = head_dim // blk_d
    reduce_cores = min(num_vec_cores, num_col_blocks)

    @T.prim_func
    def engram_sinkhorn_step_reduce_kernel_asc(
        col_sumsq_partial: T.Tensor[(num_partials, head_dim), T.float32],
        col_sumsq: T.Tensor[(head_dim,), T.float32],
    ):
        with T.Kernel(reduce_cores) as core_id:
            partial_ub = T.alloc_shared((num_partials, blk_d), T.float32)
            col_sumsq_ub = T.alloc_shared((blk_d,), T.float32)

            for col_block in T.Persistent([num_col_blocks], reduce_cores, core_id, group_size=1, num_stages=0):
                col_start = col_block * blk_d
                T.copy(col_sumsq_partial[:, col_start : col_start + blk_d], partial_ub)

                with T.SimdVF():
                    zero = S.vdup(0.0, T.float32)
                    accum = S.alloc_local((num_accs,), T.float32)
                    for acc_id in T.unroll(num_accs, explicit=True):
                        accum[acc_id] = zero

                    for partial_id in T.serial(num_partials):
                        for acc_id in T.unroll(num_accs, explicit=True):
                            offset = acc_id * vec_size
                            accum[acc_id] = S.vadd(accum[acc_id], S.vld(partial_ub[partial_id, offset]))

                    for acc_id in T.unroll(num_accs, explicit=True):
                        S.vsts(col_sumsq_ub[acc_id * vec_size], accum[acc_id])

                T.copy(col_sumsq_ub, col_sumsq[col_start : col_start + blk_d])

    return engram_sinkhorn_step_reduce_kernel_asc


@tilelang.jit(compile_flags=['--cce-disable-vf-stack-reserved-ubuf'])
def get_engram_sinkhorn_finalize_kernel_asc(
    head_dim: int,
    eps: float,
    alpha: float,
    num_vec_cores: int,
):
    vec_size = 64
    num_accs = 4 if head_dim % 256 == 0 else 2
    assert head_dim % (vec_size * num_accs) == 0
    num_rows = T.dynamic('num_rows')
    num_chunks = head_dim // vec_size
    num_stages = 4

    @T.prim_func
    def engram_sinkhorn_finalize_kernel_asc(
        matrix: T.Tensor[(num_rows, head_dim), T.float32],
        row_scale: T.Tensor[(num_rows,), T.float32],
        col_scale: T.Tensor[(head_dim,), T.float32],
        out: T.Tensor[(num_rows, head_dim), T.float32],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            col_scale_ub = T.alloc_shared((head_dim,), T.float32)
            matrix_ub = T.alloc_shared((head_dim,), T.float32)
            out_ub = T.alloc_shared((head_dim,), T.float32)
            row_scale_ub = T.alloc_shared((8,), T.float32)
            T.annotate_buffer_versions(
                {
                    matrix_ub: num_stages,
                    out_ub: num_stages,
                    row_scale_ub: num_stages,
                }
            )

            T.copy(col_scale, col_scale_ub)
            for row in T.Persistent([num_rows], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
                T.copy(matrix[row, :], matrix_ub)
                T.copy(row_scale[row : row + 1], row_scale_ub[:1])

                with T.SimdVF():
                    zero = S.vdup(0.0, T.float32)
                    alpha_vec = S.vdup(alpha, T.float32)
                    accum = S.alloc_local((num_accs,), T.float32)
                    for acc_id in T.unroll(num_accs, explicit=True):
                        accum[acc_id] = zero

                    scale = S.vld(row_scale_ub[0], dist='BRC_B32')
                    for chunk_group in T.serial(num_chunks // num_accs):
                        for acc_id in T.unroll(num_accs, explicit=True):
                            offset = (chunk_group * num_accs + acc_id) * vec_size
                            value = S.vmul(S.vmul(scale, S.vld(matrix_ub[offset])), S.vld(col_scale_ub[offset]))
                            S.vmula(accum[acc_id], value, value)
                            S.vsts(out_ub[offset], value)

                    if num_accs == 4:
                        for acc_id in T.unroll(2, explicit=True):
                            accum[acc_id] = S.vadd(accum[acc_id], accum[acc_id + 2])
                    accum[0] = S.vadd(accum[0], accum[1])

                    norm = S.vsqrt(S.vcadd(accum[0]))
                    output_scale = S.vdiv(alpha_vec, S.vadds(norm, eps))
                    output_scale_broadcast = S.vdupv(output_scale)
                    S.mem_bar('VST_VLD')
                    for chunk in T.serial(num_chunks):
                        offset = chunk * vec_size
                        S.vsts(out_ub[offset], S.vmul(S.vld(out_ub[offset]), output_scale_broadcast))

                T.copy(out_ub, out[row, :])

    return engram_sinkhorn_finalize_kernel_asc

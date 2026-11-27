import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from tile_kernels.config import get_max_ub_per_vector_core
from tile_kernels.quant.norm_forward_asc import load_f32, store_pair
from tile_kernels.utils import align


@tilelang.jit
def get_norm_backward_kernel_asc(
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
    num_vec_cores: int,
    do_recompute: bool,
    with_weight_grad: bool,
):
    num_rms_groups = hidden // rms_group_size
    vec_size = 64
    weight_grad_block_k = 128 if hidden > num_vec_cores * vec_size else vec_size
    # Reuse a matching or wider input; widening gets a separate output buffer.
    io_is_x = x_dtype == x_grad_dtype or out_grad_dtype != x_grad_dtype
    io_dtype = x_dtype if io_is_x else out_grad_dtype
    input_dtype = out_grad_dtype if io_is_x else x_dtype
    reuse_input = io_dtype.bytes >= x_grad_dtype.bytes
    # Prefetch reduction inputs first, then residual; a separate output only spans apply to store.
    row_dtypes = [io_dtype, input_dtype]
    if with_residual_out_grad:
        row_dtypes.append(x_dtype)
    if not reuse_input:
        row_dtypes.append(x_grad_dtype)
    if do_recompute and not with_residual_out_grad:
        row_dtypes.append(x_dtype)
    ub_bytes = get_max_ub_per_vector_core(use_simt=False)
    # Keep whole groups together and reserve a second copy of both reduction inputs.
    for block_g in range(num_rms_groups, 0, -1):
        if num_rms_groups % block_g:
            continue
        block_k = block_g * rms_group_size
        scratch_size = max(block_k, num_vec_cores * weight_grad_block_k)
        row_bytes = [block_k * dtype.bytes for dtype in row_dtypes]
        staged_row_bytes = sum(row_bytes)
        fixed_bytes = ((block_k + scratch_size) * T.float32.bytes + weight_grad_block_k * weight_grad_dtype.bytes) if with_weight else 0
        if fixed_bytes + staged_row_bytes + sum(row_bytes[:2]) + 3 * align(block_g, vec_size) * T.float32.bytes <= ub_bytes:
            break
    num_group_blocks = num_rms_groups // block_g
    # Shrink the tile until double buffering fits; reserve rstd per stage and one neg_c.
    num_stages = 2
    # Split group blocks use one token so their RSTD slice stays contiguous.
    block_m = vec_size if num_group_blocks == 1 else 1
    while block_m > 1 and (
        num_stages * block_m * staged_row_bytes + (num_stages + 1) * align(block_m * block_g, vec_size) * T.float32.bytes > ub_bytes - fixed_bytes
    ):
        block_m //= 2
    # Use spare UB for whole pipeline groups before assigning partial extra buffers.
    row_stats_bytes = align(block_m * block_g, vec_size) * T.float32.bytes
    stage_bytes = block_m * staged_row_bytes + row_stats_bytes
    num_stages = max(2, min(4, (ub_bytes - fixed_bytes - row_stats_bytes) // stage_bytes))
    spare_bytes = ub_bytes - fixed_bytes - block_m * staged_row_bytes - (num_stages + 1) * row_stats_bytes
    assert spare_bytes >= 0, 'norm tile exceeds the available UB even for a single stage'
    # Cover early inputs first; residual can overlap reduction and output can overlap loads.
    versions = [1] * len(row_bytes)
    for i, row_size in enumerate(row_bytes):
        tile_bytes = (num_stages - 1) * block_m * row_size
        if spare_bytes >= tile_bytes:
            versions[i] = num_stages
            spare_bytes -= tile_bytes

    num_tokens = T.dynamic('num_tokens')
    x_stride = T.dynamic('x_stride')
    out_grad_stride = T.dynamic('out_grad_stride')
    residual_out_grad_stride = T.dynamic('residual_out_grad_stride')

    @T.macro
    def reduce_tile(x_ub, weight_ub, out_grad_ub, rstd_ub, neg_c_ub, num_rows):
        # neg_c = -rstd^3 * mean(x * weighted_dy).
        with T.SimdVF():
            for row, group in T.grid(T.uint16(num_rows), T.uint16(block_g)):
                row_group = row * block_g + group
                total0, total1 = S.alloc_var(T.float32), S.alloc_var(T.float32)
                total0, total1 = S.vdup(0.0, T.float32), S.vdup(0.0, T.float32)
                for pair in T.serial(T.uint16(rms_group_size // 128)):
                    col = group * rms_group_size + pair * 128
                    x0, x1 = load_f32(x_ub[row, col]), load_f32(x_ub[row, col + 64])
                    dy0, dy1 = load_f32(out_grad_ub[row, col]), load_f32(out_grad_ub[row, col + 64])
                    if not with_weight and out_scale != 1.0:
                        dy0, dy1 = S.vmuls(dy0, out_scale), S.vmuls(dy1, out_scale)
                    if with_weight:
                        w0, w1 = load_f32(weight_ub[col]), load_f32(weight_ub[col + 64])
                        dy0, dy1 = S.vmul(dy0, w0), S.vmul(dy1, w1)
                    S.vmula(total0, x0, dy0)
                    S.vmula(total1, x1, dy1)
                S.vsts(neg_c_ub[row_group], S.vcadd(S.vadd(total0, total1)), dist='ONEPT_B32')
            # Apply RSTD to 64 statistics at once, rather than broadcasting each scalar.
            S.mem_bar('VST_VLD')
            for vector in T.serial(T.uint16(T.ceildiv(num_rows * block_g, vec_size))):
                col = vector * vec_size
                inv_rms = S.vld(rstd_ub[col])
                mean = S.vmul(S.vld(neg_c_ub[col]), S.vmuls(inv_rms, 1.0 / rms_group_size))
                S.vsts(neg_c_ub[col], S.vneg(S.vmul(S.vmul(inv_rms, inv_rms), mean)))

    @T.macro
    def apply_tile(x_ub, weight_ub, out_grad_ub, x_grad_ub, residual_ub, rstd_ub, neg_c_ub, weight_grad_ub, num_rows):
        with T.SimdVF():
            # Keep one dW pair live across rows.
            for group, pair in T.grid(T.uint16(block_g), T.uint16(rms_group_size // 128)):
                col = group * rms_group_size + pair * 128
                dw0, dw1 = S.alloc_var(T.float32), S.alloc_var(T.float32)
                if with_weight:
                    w0, w1 = load_f32(weight_ub[col]), load_f32(weight_ub[col + 64])
                if with_weight_grad:
                    dw0, dw1 = load_f32(weight_grad_ub[col]), load_f32(weight_grad_ub[col + 64])
                for row in T.serial(T.uint16(num_rows)):
                    inv_rms = S.vld(rstd_ub[row * block_g + group], dist='BRC_B32')
                    neg_c = S.vld(neg_c_ub[row * block_g + group], dist='BRC_B32')
                    x0, x1 = load_f32(x_ub[row, col]), load_f32(x_ub[row, col + 64])
                    dy0, dy1 = load_f32(out_grad_ub[row, col]), load_f32(out_grad_ub[row, col + 64])
                    if not with_weight and out_scale != 1.0:
                        dy0, dy1 = S.vmuls(dy0, out_scale), S.vmuls(dy1, out_scale)
                    if with_weight_grad:
                        dw_scale = S.vmuls(inv_rms, out_scale) if out_scale != 1.0 else inv_rms
                        S.vmula(dw0, S.vmul(x0, dy0), dw_scale)
                        S.vmula(dw1, S.vmul(x1, dy1), dw_scale)
                    if with_weight:
                        dy0, dy1 = S.vmul(dy0, w0), S.vmul(dy1, w1)
                    dx0, dx1 = S.alloc_var(T.float32), S.alloc_var(T.float32)
                    dx0, dx1 = S.vmul(dy0, inv_rms), S.vmul(dy1, inv_rms)
                    S.vmula(dx0, x0, neg_c)
                    S.vmula(dx1, x1, neg_c)
                    if with_residual_out_grad:
                        dr0, dr1 = load_f32(residual_ub[row, col]), load_f32(residual_ub[row, col + 64])
                        dx0, dx1 = S.vadd(dx0, dr0), S.vadd(dx1, dr1)
                    if do_recompute:
                        y0, y1 = S.vmul(x0, inv_rms), S.vmul(x1, inv_rms)
                        if with_weight:
                            y0, y1 = S.vmul(y0, w0), S.vmul(y1, w1)
                        elif out_scale != 1.0:
                            y0, y1 = S.vmuls(y0, out_scale), S.vmuls(y1, out_scale)
                        store_pair(residual_ub[row, col], residual_ub[row, col + 64], y0, y1)
                    store_pair(x_grad_ub[row, col], x_grad_ub[row, col + 64], dx0, dx1)
                if with_weight_grad:
                    store_pair(weight_grad_ub[col], weight_grad_ub[col + 64], dw0, dw1)

    @T.macro
    def reduce_weight_grad(partial_ub, out_ub):
        with T.SimdVF():
            for chunk in T.serial(T.uint16(weight_grad_block_k // vec_size)):
                total = S.alloc_var(T.float32)
                total = S.vdup(0.0, T.float32)
                for core in T.serial(T.uint16(num_vec_cores)):
                    total = S.vadd(total, S.vld(partial_ub[core, chunk * vec_size]))
                if weight_grad_initialized:
                    total = S.vadd(total, load_f32(out_ub[chunk * vec_size]))
                value = S.vcvt(total, T.bfloat16) if weight_grad_dtype == T.bfloat16 else total
                S.vsts(out_ub[chunk * vec_size], value, dist='PK_B32' if weight_grad_dtype == T.bfloat16 else 'NORM_B32')

    @T.prim_func
    def norm_backward_kernel(
        x: T.StridedTensor[(num_tokens, hidden), (x_stride, 1), x_dtype],
        weight: T.Tensor[(hidden,), weight_dtype],
        out_grad: T.StridedTensor[(num_tokens, hidden), (out_grad_stride, 1), out_grad_dtype],
        rstd: T.Tensor[(num_tokens * num_rms_groups,), T.float32],
        x_grad: T.Tensor[(num_tokens, hidden), x_grad_dtype],
        weight_grad_partial: T.Tensor[(num_vec_cores, hidden), T.float32],
        weight_grad: T.Tensor[(hidden,), weight_grad_dtype],
        residual_out_grad: T.StridedTensor[(num_tokens, hidden), (residual_out_grad_stride, 1), x_dtype],
        out: T.Tensor[(num_tokens, hidden), x_dtype],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            io_ub = T.alloc_shared((block_m, block_k), io_dtype)
            input_ub = T.alloc_shared((block_m, block_k), input_dtype)
            x_ub = io_ub if io_is_x else input_ub
            out_grad_ub = input_ub if io_is_x else io_ub
            if reuse_input:
                x_grad_ub = T.view(io_ub, (block_m, block_k * io_dtype.bytes // x_grad_dtype.bytes), x_grad_dtype)
            else:
                x_grad_ub = T.alloc_shared((block_m, block_k), x_grad_dtype)
            weight_ub = T.alloc_shared((block_k,), T.float32)
            # Recompute reuses residual storage after its gradient is consumed.
            residual_ub = T.alloc_shared((block_m, block_k), x_dtype)
            scratch_ub = T.alloc_shared((scratch_size,), T.float32)
            weight_grad_ub = T.Tensor((block_k,), T.float32, scratch_ub.data, scope='shared')
            rstd_ub = T.alloc_shared((row_stats_bytes // T.float32.bytes,), T.float32)
            neg_c_ub = T.alloc_shared((row_stats_bytes // T.float32.bytes,), T.float32)
            buffer_versions = {io_ub: versions[0], input_ub: versions[1], rstd_ub: num_stages, neg_c_ub: 1}
            if with_residual_out_grad:
                buffer_versions[residual_ub] = versions[2]
            if not reuse_input:
                buffer_versions[x_grad_ub] = versions[2 + with_residual_out_grad]
            if do_recompute and not with_residual_out_grad:
                buffer_versions[residual_ub] = versions[-1]
            T.annotate_buffer_versions(buffer_versions)
            io_src = x if io_is_x else out_grad
            input_src = out_grad if io_is_x else x
            weight_grad_out_ub = T.alloc_shared((weight_grad_block_k,), weight_grad_dtype)
            for group_block in T.serial(num_group_blocks):
                col_start = group_block * block_k
                if with_weight:
                    # Reuse dW scratch only when weight needs widening.
                    weight_raw_ub = T.Tensor((block_k,), weight_dtype, scratch_ub.data, scope='shared') if weight_dtype != T.float32 else weight_ub
                    T.copy(weight[col_start], weight_raw_ub)
                    if weight_dtype != T.float32 or out_scale != 1.0:
                        with T.SimdVF():
                            for pair in T.serial(T.uint16(block_k // 128)):
                                col = pair * 128
                                w0, w1 = load_f32(weight_raw_ub[col]), load_f32(weight_raw_ub[col + 64])
                                if out_scale != 1.0:
                                    w0, w1 = S.vmuls(w0, out_scale), S.vmuls(w1, out_scale)
                                S.vsts(weight_ub[col], w0)
                                S.vsts(weight_ub[col + 64], w1)
                if with_weight_grad:
                    with T.SimdVF():
                        T.clear(weight_grad_ub)

                for tile in T.Pipelined(T.ceildiv(num_tokens, block_m * num_vec_cores), num_stages=num_stages):
                    token_start = (tile * num_vec_cores + core_id) * block_m
                    if token_start < num_tokens:
                        with T.Stage(0):
                            T.copy(rstd[token_start * num_rms_groups + group_block * block_g], rstd_ub)
                            # Prefetch before waiting on single-buffer inputs; IO is released last at equal depth.
                            for depth in T.unroll(num_stages, 0, -1, explicit=True):
                                if versions[1] == depth:
                                    T.copy(input_src[token_start, col_start], input_ub)
                                if versions[0] == depth:
                                    T.copy(io_src[token_start, col_start], io_ub)
                        reduce_tile(x_ub, weight_ub, out_grad_ub, rstd_ub, neg_c_ub, T.min(block_m, num_tokens - token_start))
                        # Only apply waits for residual; its DMA can overlap the reduction.
                        if with_residual_out_grad:
                            T.copy(residual_out_grad[token_start, col_start], residual_ub)
                        apply_tile(
                            x_ub,
                            weight_ub,
                            out_grad_ub,
                            x_grad_ub,
                            residual_ub,
                            rstd_ub,
                            neg_c_ub,
                            weight_grad_ub,
                            T.min(block_m, num_tokens - token_start),
                        )
                        T.copy(x_grad_ub[:, :block_k], x_grad[token_start, col_start])
                        if do_recompute:
                            T.copy(residual_ub, out[token_start, col_start])
                if with_weight_grad:
                    T.copy(weight_grad_ub, weight_grad_partial[core_id, col_start])
            if with_weight_grad:
                T.ascend_sync_inter_arrive('PIPE_MTE3', 0)
                T.ascend_sync_inter_wait('PIPE_MTE2', 0)
                partial_ub = T.reshape(scratch_ub, (scratch_size // weight_grad_block_k, weight_grad_block_k))
                for pid_hidden in T.Persistent([hidden // weight_grad_block_k], num_vec_cores, core_id, group_size=1, num_stages=1):
                    col = pid_hidden * weight_grad_block_k
                    T.copy(weight_grad_partial[0, col], partial_ub[:num_vec_cores, :])
                    if weight_grad_initialized:
                        T.copy(weight_grad[col], weight_grad_out_ub)
                    reduce_weight_grad(partial_ub, weight_grad_out_ub)
                    T.copy(weight_grad_out_ub, weight_grad[col])

    return norm_backward_kernel

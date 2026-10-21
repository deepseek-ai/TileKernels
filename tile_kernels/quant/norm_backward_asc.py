import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from tile_kernels.config import get_max_ub_per_vector_core
from tile_kernels.quant.norm_forward_asc import load_f32, load_f32_pair, store_pair
from tile_kernels.utils import align


@tilelang.jit
def get_norm_backward_kernel_asc(
    hidden: int,
    with_weight: bool,
    with_residual_out_grad: bool,
    weight_grad_initialized: bool,
    out_scale: float,
    x_dtype: T.dtype,
    weight_grad_dtype: T.dtype,
    num_vec_cores: int,
):
    assert hidden % 128 == 0
    vec_size = 64
    row_bytes = hidden * x_dtype.bytes
    # Target 16 KiB per input tile; a wide row is the minimum tile.
    block_m = min(vec_size, 1 << max(0, (16384 // row_bytes).bit_length() - 1))
    block_k = 128 if hidden > num_vec_cores * vec_size else vec_size
    scratch_size = max(hidden, num_vec_cores * block_k)

    row_stats_bytes = align(block_m * T.float32.bytes, 32)
    fixed_bytes = (row_bytes + scratch_size * T.float32.bytes) * with_weight + row_stats_bytes
    stage_bytes = 3 * block_m * row_bytes + row_stats_bytes
    num_stages = min(4, (get_max_ub_per_vector_core(use_simt=False) - fixed_bytes) // stage_bytes)
    assert num_stages >= 2, 'norm tile exceeds the available UB for two stages'

    num_tokens = T.dynamic('num_tokens')
    x_stride = T.dynamic('x_stride')
    out_grad_stride = T.dynamic('out_grad_stride')
    residual_out_grad_stride = T.dynamic('residual_out_grad_stride')

    @T.macro
    def load_scaled_grad_pair(buffer, col):
        dy0, dy1 = load_f32_pair(buffer[col], buffer[col + 64])
        if out_scale != 1.0:
            dy0, dy1 = S.vmuls(dy0, out_scale), S.vmuls(dy1, out_scale)
        return dy0, dy1

    @T.macro
    def backward_tile(x_ub, weight_ub, out_grad_ub, x_grad_ub, rstd_ub, mean_ub, weight_grad_ub, num_rows):
        x_ub = T.reshape(x_ub, (block_m * hidden,))
        out_grad_ub = T.reshape(out_grad_ub, (block_m * hidden,))
        x_grad_ub = T.reshape(x_grad_ub, (block_m * hidden,))
        with T.SimdVF():
            # g = dy * out_scale * weight; dX = (g - xhat * mean(xhat * g)) * rstd.
            for row in T.serial(T.uint16(num_rows)):
                rstd_value = S.vld(rstd_ub[row], dist='BRC_B32')
                total0, total1 = S.alloc_var(T.float32), S.alloc_var(T.float32)
                total0, total1 = S.vdup(0.0, T.float32), S.vdup(0.0, T.float32)
                for pair in T.serial(T.uint16(hidden // 128)):
                    col = pair * 128
                    x0, x1 = load_f32_pair(x_ub[row * hidden + col], x_ub[row * hidden + col + 64])
                    dy0, dy1 = load_scaled_grad_pair(out_grad_ub, row * hidden + col)
                    if with_weight:
                        w0, w1 = load_f32_pair(weight_ub[col], weight_ub[col + 64])
                        dy0, dy1 = S.vmul(dy0, w0), S.vmul(dy1, w1)
                    S.vmula(total0, S.vmul(x0, rstd_value), dy0)
                    S.vmula(total1, S.vmul(x1, rstd_value), dy1)
                mean_value = S.vmuls(S.vcadd(S.vadd(total0, total1)), 1.0 / hidden)
                S.vsts(mean_ub[row], mean_value, dist='ONEPT_B32')
            S.mem_bar('VST_VLD')
            # Keep one native-layout dW pair live across rows; canonicalize once per core below.
            for pair in T.serial(T.uint16(hidden // 128)):
                col = pair * 128
                dw0, dw1 = S.alloc_var(T.float32), S.alloc_var(T.float32)
                if with_weight:
                    w0, w1 = load_f32_pair(weight_ub[col], weight_ub[col + 64])
                    dw0, dw1 = load_f32_pair(weight_grad_ub[col], weight_grad_ub[col + 64])
                for row in T.serial(T.uint16(num_rows)):
                    inv_rms = S.vld(rstd_ub[row], dist='BRC_B32')
                    mean = S.vld(mean_ub[row], dist='BRC_B32')
                    x0, x1 = load_f32_pair(x_ub[row * hidden + col], x_ub[row * hidden + col + 64])
                    dy0, dy1 = load_scaled_grad_pair(out_grad_ub, row * hidden + col)
                    x0, x1 = S.vmul(x0, inv_rms), S.vmul(x1, inv_rms)
                    if with_weight:
                        S.vmula(dw0, x0, dy0)
                        S.vmula(dw1, x1, dy1)
                        dy0, dy1 = S.vmul(dy0, w0), S.vmul(dy1, w1)
                    dx0 = S.vmul(S.vsub(dy0, S.vmul(x0, mean)), inv_rms)
                    dx1 = S.vmul(S.vsub(dy1, S.vmul(x1, mean)), inv_rms)
                    if with_residual_out_grad:
                        dr0, dr1 = load_f32_pair(x_grad_ub[row * hidden + col], x_grad_ub[row * hidden + col + 64])
                        dx0, dx1 = S.vadd(dx0, dr0), S.vadd(dx1, dr1)
                    store_pair(x_grad_ub[row * hidden + col], x_grad_ub[row * hidden + col + 64], dx0, dx1)
                if with_weight:
                    store_pair(weight_grad_ub[col], weight_grad_ub[col + 64], dw0, dw1)

    @T.macro
    def reduce_weight_grad(partial_ub, out_ub):
        for chunk in T.serial(T.uint16(block_k // vec_size)):
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
        weight: T.Tensor[(hidden,), x_dtype],
        out_grad: T.StridedTensor[(num_tokens, hidden), (out_grad_stride, 1), x_dtype],
        rstd: T.Tensor[(num_tokens,), T.float32],
        x_grad: T.Tensor[(num_tokens, hidden), x_dtype],
        weight_grad_partial: T.Tensor[(num_vec_cores, hidden), T.float32],
        residual_out_grad: T.StridedTensor[(num_tokens, hidden), (residual_out_grad_stride, 1), x_dtype],
        weight_grad: T.Tensor[(hidden,), weight_grad_dtype],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            x_ub = T.alloc_shared((block_m, hidden), x_dtype)
            weight_ub = T.alloc_shared((hidden,), x_dtype)
            out_grad_ub = T.alloc_shared((block_m, hidden), x_dtype)
            x_grad_ub = T.alloc_shared((block_m, hidden), x_dtype)
            scratch_ub = T.alloc_shared((scratch_size,), T.float32)
            weight_grad_ub = T.Tensor((hidden,), T.float32, scratch_ub.data, scope='shared')
            rstd_ub = T.alloc_shared((block_m,), T.float32)
            mean_ub = T.alloc_shared((block_m,), T.float32)
            T.annotate_buffer_versions(
                {
                    x_ub: num_stages,
                    out_grad_ub: num_stages,
                    x_grad_ub: num_stages,
                    rstd_ub: num_stages,
                    mean_ub: 1,
                }
            )
            if with_weight:
                T.copy(weight, weight_ub)
                with T.SimdVF():
                    T.clear(weight_grad_ub)

            for tile in T.Pipelined(T.ceildiv(num_tokens, block_m * num_vec_cores), num_stages=num_stages):
                token_start = (tile * num_vec_cores + core_id) * block_m
                if token_start < num_tokens:
                    T.copy(rstd[token_start], rstd_ub)
                    T.copy(x[token_start, 0], x_ub)
                    T.copy(out_grad[token_start, 0], out_grad_ub)
                    if with_residual_out_grad:
                        T.copy(residual_out_grad[token_start, 0], x_grad_ub)
                    backward_tile(
                        x_ub,
                        weight_ub,
                        out_grad_ub,
                        x_grad_ub,
                        rstd_ub,
                        mean_ub,
                        weight_grad_ub,
                        T.min(block_m, num_tokens - token_start),
                    )
                    T.copy(x_grad_ub, x_grad[token_start, 0])
            if with_weight:
                if x_dtype == T.bfloat16:
                    with T.SimdVF():
                        for pair in T.serial(T.uint16(hidden // 128)):
                            partial0, partial1 = load_f32_pair(weight_grad_ub[pair * 128], weight_grad_ub[pair * 128 + 64])
                            partial0, partial1 = S.vintlv(partial0, partial1)
                            store_pair(weight_grad_ub[pair * 128], weight_grad_ub[pair * 128 + 64], partial0, partial1)
                T.copy(weight_grad_ub, weight_grad_partial[core_id, 0])
                T.ascend_sync_inter_arrive('PIPE_MTE3', 0)
                T.ascend_sync_inter_wait('PIPE_MTE2', 0)
                partial_ub = T.reshape(scratch_ub, (scratch_size // block_k, block_k))
                out_ub = T.view(weight_ub, (hidden * x_dtype.bytes // weight_grad_dtype.bytes,), weight_grad_dtype)
                for pid_hidden in T.Persistent([hidden // block_k], num_vec_cores, core_id, group_size=1, num_stages=1):
                    col = pid_hidden * block_k
                    T.copy(weight_grad_partial[0, col], partial_ub[:num_vec_cores, :block_k])
                    if weight_grad_initialized:
                        T.copy(weight_grad[col], out_ub[:block_k])
                    with T.SimdVF():
                        reduce_weight_grad(partial_ub, out_ub)
                    T.copy(out_ub[:block_k], weight_grad[col])

    return norm_backward_kernel

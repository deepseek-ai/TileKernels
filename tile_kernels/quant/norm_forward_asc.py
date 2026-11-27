import math

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from tile_kernels.config import get_max_ub_per_vector_core
from tile_kernels.quant.common import CastOutputConfig, get_sf_shape, get_sf_and_inv_asc
from tile_kernels.quant.rstd import make_rstd_impl
from tile_kernels.utils import align, ceil_div


@T.macro
def load_f32(src: T.Ref):
    """Load 64 contiguous elements as FP32."""
    if src.dtype == T.bfloat16:
        return S.vcvt(S.vld(src, dist='UNPK_B16'), T.float32, part=0)
    return S.vld(src)


@T.macro
def store_pair(dst0: T.Ref, dst1: T.Ref, x0, x1):
    """Store two contiguous vectors, converting to the output dtype."""
    if dst0.dtype == T.bfloat16:
        S.vsts(dst0, S.vcvt(x0, T.bfloat16), dist='PK_B32')
        S.vsts(dst1, S.vcvt(x1, T.bfloat16), dist='PK_B32')
    else:
        S.vsts(dst0, x0)
        S.vsts(dst1, x1)


@tilelang.jit
def get_norm_forward_kernel_asc(
    hidden: int,
    rms_group_size: int,
    with_weight: bool,
    with_residual: bool,
    eps: float,
    out_scale: float,
    x_dtype: T.dtype,
    weight_dtype: T.dtype,
    out_config: CastOutputConfig,
    num_vec_cores: int,
    with_out_bf16: bool = False,
    with_rstd: bool = False,
):
    norm_dtype = x_dtype if out_config.with_sf else out_config.dtype
    num_rms_groups = hidden // rms_group_size
    # Keep complete groups in a block, targeting at most 32 KiB per tensor.
    block_g = max(
        g
        for g in range(1, num_rms_groups + 1)
        if num_rms_groups % g == 0 and (g == 1 or g * rms_group_size * max(x_dtype.bytes, weight_dtype.bytes, norm_dtype.bytes) <= 32768)
    )
    num_group_blocks = num_rms_groups // block_g
    block_k = block_g * rms_group_size
    block_k_aligned = align(block_k, 256) if out_config.with_sf else block_k
    row_bytes = block_k_aligned * x_dtype.bytes
    vec_size = 64
    load_size = 256 // x_dtype.bytes
    num_per_channels = out_config.sf_block[1]
    use_bf16_quant = x_dtype == T.bfloat16 and out_config.round_sf
    num_parts = load_size // vec_size
    num_accumulators = min(4, rms_group_size // vec_size)
    reduction_cols = num_accumulators * vec_size

    # SF layout only affects stores. All formats share normalization and quantization.
    transpose_sf = out_config.with_sf and out_config.use_tma_aligned_col_major_sf
    num_groups = block_k // num_per_channels
    sf_size = align(num_groups, vec_size)
    # Transpose SF as raw bits: pairs of exponent bytes or individual FP32 scales.
    sf_word_dtype = T.uint16 if out_config.use_packed_ue8m0 else T.uint32
    pack_factor = sf_word_dtype.bytes // out_config.sf_dtype.bytes
    sf_lanes = 256 // sf_word_dtype.bytes
    sf_store_dist = f'NORM_B{sf_word_dtype.bits}'
    num_sf_words = ceil_div(num_groups, pack_factor)

    # Target 32 KiB of input and residual per tile.
    tile_capacity = max(1, 32768 // (1 + with_residual) // row_bytes)
    min_rows_per_block = 8 if transpose_sf else 4
    # Amortize RSTD stores over up to 128 rows and 4096 statistics.
    max_rows_per_block = 16 if transpose_sf else max(min_rows_per_block, min(128, 1 << max(0, (4096 // block_g).bit_length() - 1)))
    block_m = min(max_rows_per_block, vec_size, 1 << (tile_capacity.bit_length() - 1))
    rstds_per_tile = block_m * block_g
    rstd_offset_span = 0 if not with_rstd else vec_size - math.gcd(rstds_per_tile, vec_size)
    num_rstd_vecs = ceil_div(rstds_per_tile + rstd_offset_span, vec_size)
    sf_rows = max(max_rows_per_block, 32 // sf_word_dtype.bytes) if transpose_sf else block_m
    rstd_out_size = align(((max_rows_per_block - block_m) * block_g if with_rstd else 0) + num_rstd_vecs * vec_size, vec_size)

    tile_bytes = block_m * row_bytes
    out_tile_bytes = block_m * block_k_aligned * norm_dtype.bytes
    stage_bytes = (
        (1 + with_residual) * tile_bytes  # x_ub, residual_ub
        + out_tile_bytes  # out_ub
        + (block_m * block_k_aligned * out_config.dtype.bytes if out_config.with_sf else 0)  # quantized_ub
        + (block_m * sf_size * out_config.sf_dtype.bytes if out_config.with_sf and not transpose_sf else 0)  # scales_ub
    )
    fixed_bytes = (
        (block_k_aligned * weight_dtype.bytes if with_weight else 0)  # weight_ub
        + rstd_out_size * T.float32.bytes  # rstd_ub
        + (block_m * sf_size * T.float32.bytes if out_config.with_sf else 0)  # sf_scratch_ub
        + (sf_rows * sf_size * out_config.sf_dtype.bytes if transpose_sf else 0)  # scales_ub
        + (sf_size // pack_factor * sf_rows * sf_word_dtype.bytes if transpose_sf else 0)  # transposed_sf_ub
        + (sf_lanes * sf_word_dtype.bytes if transpose_sf else 0)  # transpose_indices_ub
    )
    available_bytes = get_max_ub_per_vector_core(use_simt=False) - fixed_bytes
    num_stages = min(4, available_bytes // stage_bytes)
    # Reuse the input buffer for the output when it buys a deeper pipeline.
    reused_stages = min(4, available_bytes // (stage_bytes - out_tile_bytes))
    reuse_input = norm_dtype == x_dtype and num_stages < 3 and reused_stages > num_stages
    if reuse_input:
        num_stages = reused_stages
    assert num_stages > 0, 'norm tile exceeds the available UB'

    num_tokens = T.dynamic('num_tokens')
    batch = T.dynamic('batch')
    x_stride = T.dynamic('x_stride')
    out_stride = T.dynamic('out_stride')
    x_batch_stride = T.dynamic('x_batch_stride')
    out_batch_stride = T.dynamic('out_batch_stride')
    residual_stride = T.dynamic('residual_stride')
    residual_batch_stride = T.dynamic('residual_batch_stride')
    residual_out_stride = T.dynamic('residual_out_stride')
    residual_out_batch_stride = T.dynamic('residual_out_batch_stride')

    out_bf16_stride = T.dynamic('out_bf16_stride')
    out_bf16_batch_stride = T.dynamic('out_bf16_batch_stride')
    out_sf_stride = T.dynamic('out_sf_stride')
    out_sf_batch_stride = T.dynamic('out_sf_batch_stride')
    sf_shape = get_sf_shape((num_tokens, hidden), out_config) if out_config.with_sf else (1, 1)

    @T.macro
    def clear_padding(buffer):
        # Clear the tail vectors excluded from the DMA copy.
        zero = S.vdup(0.0, buffer.dtype)
        for vector in T.serial(T.uint16(block_k // load_size), T.uint16(block_k_aligned // load_size)):
            for row in T.serial(T.uint16(block_m)):
                S.vsts(buffer[row, vector * load_size], zero)

    compute_rstd = make_rstd_impl(rms_group_size, eps)

    @T.macro
    def reduce_group(x_ub, residual_ub, row, col_base):
        sums = S.alloc_local((num_accumulators,), T.float32)
        for i in T.unroll(num_accumulators, explicit=True):
            sums[i] = S.vdup(0.0, T.float32)
        for step in T.serial(T.uint16(T.ceildiv(rms_group_size, reduction_cols))):
            for chunk in T.unroll(reduction_cols // load_size, explicit=True):
                col = step * reduction_cols + chunk * load_size
                if col < rms_group_size:
                    x = S.alloc_var(x_dtype)
                    x = S.vld(x_ub[row, col_base + col])
                    if with_residual:
                        x = S.vadd(x, S.vld(residual_ub[row, col_base + col]))
                        S.vsts(residual_ub[row, col_base + col], x)
                    low = S.vcvt(x, T.float32, part=0) if x_dtype == T.bfloat16 else x
                    if rms_group_size <= reduction_cols:
                        sums[chunk * num_parts] = S.vmul(low, low)
                    else:
                        S.vmula(sums[chunk * num_parts], low, low)
                    if x_ub.dtype == T.bfloat16:
                        high = S.vcvt(x, T.float32, part=1)
                        if rms_group_size <= reduction_cols:
                            sums[chunk * num_parts + 1] = S.vmul(high, high)
                        else:
                            S.vmula(sums[chunk * num_parts + 1], high, high)
        for i in T.unroll(1, num_accumulators, explicit=True):
            sums[0] = S.vadd(sums[0], sums[i])
        return S.vcadd(sums[0])

    @T.macro
    def apply_tile(x_ub, weight_ub, out_ub, rstd_ub, rstd_base):
        for group in T.serial(T.uint16(block_g)):
            for pair in T.serial(T.uint16(rms_group_size // 128)):
                col = group * rms_group_size + pair * 128
                if with_weight:
                    w0, w1 = (load_f32(weight_ub[col]), load_f32(weight_ub[col + 64]))
                    if out_scale != 1.0:
                        w0, w1 = S.vmuls(w0, out_scale), S.vmuls(w1, out_scale)
                for row in T.serial(T.uint16(block_m)):
                    inv_rms = S.vld(rstd_ub[rstd_base + row * block_g + group], dist='BRC_B32')
                    x0, x1 = (load_f32(x_ub[row, col]), load_f32(x_ub[row, col + 64]))
                    y0, y1 = S.vmul(x0, inv_rms), S.vmul(x1, inv_rms)
                    if with_weight:
                        y0, y1 = S.vmul(y0, w0), S.vmul(y1, w1)
                    elif out_scale != 1.0:
                        y0, y1 = S.vmuls(y0, out_scale), S.vmuls(y1, out_scale)
                    store_pair(out_ub[row, col], out_ub[row, col + 64], y0, y1)

    @T.macro
    def quantize_tile(source, quantized, auxiliary, scratch, scales, scale_row_base):
        inverse_scales = T.Tensor((block_m, sf_size), T.bfloat16 if use_bf16_quant else T.float32, scratch.data)
        with T.SimdVF():
            low_mask = S.pset(T.float32.bits, 'PAT_VL32')
            store_mask = S.pset(32, 'PAT_VL8')
            for row in T.serial(T.uint16(block_m)):
                for tile in T.serial(T.uint16(block_k_aligned // 256)):
                    col = tile * 256
                    if x_dtype == T.bfloat16:
                        abs_mask = S.vdup(0x7FFF, T.uint16)
                        even, odd = S.vld2(source[row, col], dist='DINTLV_B16')
                        pairs = S.vmax(
                            S.vand(T.reinterpret(even, 'uint16x128'), abs_mask),
                            S.vand(T.reinterpret(odd, 'uint16x128'), abs_mask),
                        )
                        maxima = S.vcgmax(pairs)
                        dense, _ = S.vintlv(S.vdup(0.0, T.bfloat16), T.reinterpret(maxima, 'bfloat16x128'))
                        amax = T.reinterpret(dense, 'float32x64')
                    else:
                        even0, odd0 = S.vld2(source[row, col], dist='DINTLV_B32')
                        even1, odd1 = S.vld2(source[row, col + 128], dist='DINTLV_B32')
                        pairs0 = S.vmax(S.vabs(even0), S.vabs(odd0))
                        pairs1 = S.vmax(S.vabs(even1), S.vabs(odd1))
                        even, odd = S.vdintlv(pairs0, pairs1)
                        amax = S.vcgmax(S.vmax(even, odd))
                    S.vsts(scratch[row, tile * 8], amax, store_mask)
            S.mem_bar('VST_VLD')
            for row in T.serial(T.uint16(block_m)):
                for group_tile in T.serial(T.uint16(sf_size // vec_size)):
                    group = group_tile * vec_size
                    sf, inv = get_sf_and_inv_asc(S.vld(scratch[row, group]), out_config)
                    if use_bf16_quant:
                        S.vsts(inverse_scales[row, group], S.vcvt(inv, T.bfloat16), dist='PK_B32')
                    else:
                        S.vsts(inverse_scales[row, group], inv)
                    if out_config.use_packed_ue8m0:
                        S.vsts(scales[scale_row_base + row, group], T.reinterpret(sf, 'uint8x256'), dist='PK4_B32')
                    else:
                        S.vsts(scales[scale_row_base + row, group], sf)
            S.mem_bar('VST_VLD')
            if use_bf16_quant:
                # Power-of-two scaling is exact in BF16 for every nonzero FP8 result.
                for row in T.serial(T.uint16(block_m)):
                    for tile in T.serial(T.uint16(block_k_aligned // 256)):
                        col = tile * 256
                        even, odd = S.vld2(source[row, col], dist='DINTLV_B16')
                        inverse = S.vld(inverse_scales[row, tile * 8], dist='E2B_B16')
                        even, odd = S.vmul(even, inverse), S.vmul(odd, inverse)
                        parts = S.alloc_local((4,), out_config.dtype)
                        parts[0] = S.vcvt(S.vcvt(even, T.float32, part=0), out_config.dtype, part=0)
                        parts[2] = S.vcvt(S.vcvt(even, T.float32, part=1), out_config.dtype, part=2)
                        parts[1] = S.vcvt(S.vcvt(odd, T.float32, part=0), out_config.dtype, part=1)
                        parts[3] = S.vcvt(S.vcvt(odd, T.float32, part=1), out_config.dtype, part=3)
                        even_bytes = S.vor(T.reinterpret(parts[0], 'uint8x256'), T.reinterpret(parts[2], 'uint8x256'))
                        odd_bytes = S.vor(T.reinterpret(parts[1], 'uint8x256'), T.reinterpret(parts[3], 'uint8x256'))
                        S.vsts(quantized[row, col], T.reinterpret(S.vor(even_bytes, odd_bytes), 'float8_e4m3fnx256'), dist='NORM_B8')
            else:
                for row in T.serial(T.uint16(block_m)):
                    for vector in T.serial(T.uint16(block_k_aligned // vec_size)):
                        col = vector * vec_size
                        group = vector * 2
                        inv0 = S.vld(scratch[row, group], dist='BRC_B32')
                        inv1 = S.vld(scratch[row, group + 1], dist='BRC_B32')
                        inv = S.vsel(inv0, inv1, low_mask)
                        value = load_f32(source[row, col])
                        if with_out_bf16 and x_dtype != T.bfloat16:
                            S.vsts(auxiliary[row, col], S.vcvt(value, T.bfloat16), dist='PK_B32')
                        result = S.vcvt(S.vmul(value, inv), out_config.dtype)
                        S.vsts(quantized[row, col], result, dist='PK4_B32')

    @T.prim_func
    def norm_forward_kernel(
        x: T.StridedTensor[(batch, num_tokens, hidden), (x_batch_stride, x_stride, 1), x_dtype],
        out: T.StridedTensor[(batch, num_tokens, hidden), (out_batch_stride, out_stride, 1), out_config.dtype],
        out_sf: T.StridedTensor[(batch, *sf_shape), (out_sf_batch_stride, out_sf_stride, 1), out_config.sf_dtype],
        out_bf16: T.StridedTensor[(batch, num_tokens, hidden), (out_bf16_batch_stride, out_bf16_stride, 1), T.bfloat16],
        weight: T.Tensor[(hidden,), weight_dtype],
        rstd: T.Tensor[(batch, num_tokens * num_rms_groups), T.float32],
        residual: T.StridedTensor[(batch, num_tokens, hidden), (residual_batch_stride, residual_stride, 1), x_dtype],
        residual_out: T.StridedTensor[(batch, num_tokens, hidden), (residual_out_batch_stride, residual_out_stride, 1), x_dtype],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            x_ub = T.alloc_shared((block_m, block_k_aligned), x_dtype)
            out_ub = x_ub if reuse_input else T.alloc_shared((block_m, block_k_aligned), norm_dtype)
            weight_ub = T.alloc_shared((block_k_aligned,), weight_dtype)
            residual_ub = T.alloc_shared((block_m, block_k_aligned), x_dtype)
            # Quantization consumes FP32 in order, so its BF16 auxiliary can compact it in place.
            out_bf16_ub = T.Tensor((block_m, block_k_aligned), T.bfloat16, out_ub.data, scope='shared')
            rstd_ub = T.alloc_shared((rstd_out_size,), T.float32)
            quantized_ub = T.alloc_shared((block_m, block_k_aligned), out_config.dtype)
            sf_scratch_ub = T.alloc_shared((block_m, sf_size), T.float32)
            scales_ub = T.alloc_shared((sf_rows, sf_size), out_config.sf_dtype)
            if transpose_sf:
                transposed_sf_ub = T.alloc_shared((sf_size // pack_factor, sf_rows), sf_word_dtype)
                transpose_indices_ub = T.alloc_shared((sf_lanes,), sf_word_dtype)
                scale_words = T.Tensor((sf_rows, sf_size // pack_factor), sf_word_dtype, scales_ub.data)
                out_sf_words = T.StridedTensor(
                    (batch, ceil_div(hidden // num_per_channels, pack_factor), num_tokens),
                    (out_sf_batch_stride // pack_factor, out_sf_stride // pack_factor, 1),
                    sf_word_dtype,
                    data=out_sf.data,
                )
            T.annotate_buffer_versions(
                {
                    x_ub: num_stages,
                    out_ub: num_stages,
                    out_bf16_ub: num_stages,
                    residual_ub: num_stages,
                    quantized_ub: num_stages,
                    scales_ub: 1 if transpose_sf else num_stages,
                    rstd_ub: 1,
                    sf_scratch_ub: 1,
                }
            )
            if transpose_sf:
                with T.SimdVF():
                    lane = T.reinterpret(S.vci(0, f'int{sf_word_dtype.bits}'), f'{sf_word_dtype}x{sf_lanes}')
                    row = S.vand(lane, S.vdup(sf_rows - 1, sf_word_dtype))
                    pack = S.vshrs(lane, sf_rows.bit_length() - 1)
                    S.vsts(transpose_indices_ub[0], S.vadd(S.vmuls(row, sf_size // pack_factor), pack), dist=sf_store_dist)
            rows_per_block = T.alloc_var(
                T.int32, init=T.clamp(T.ceildiv(num_tokens, num_vec_cores * block_m) * block_m, min_rows_per_block, max_rows_per_block)
            )
            num_token_blocks = T.ceildiv(num_tokens, rows_per_block)
            rstd_groups = T.reshape(rstd, (batch, num_tokens, num_rms_groups))
            rstd_rows = T.Tensor((max_rows_per_block, block_g), T.float32, rstd_ub.data, scope='shared')
            for group_block in T.serial(num_group_blocks):
                col_start = group_block * block_k
                if with_weight:
                    T.copy(weight[col_start], weight_ub[:block_k])
                for batch_id, pid_token in T.Persistent([batch, num_token_blocks], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
                    token_base = pid_token * rows_per_block
                    for tile_index in T.serial(rows_per_block // block_m):
                        row_base = tile_index * block_m
                        token_start = token_base + row_base
                        if token_start < num_tokens:
                            T.copy(x[batch_id, token_start, col_start], x_ub[:, :block_k])
                            if with_residual:
                                T.copy(residual[batch_id, token_start, col_start], residual_ub[:, :block_k])
                            with T.SimdVF():
                                if block_k != block_k_aligned:
                                    clear_padding(x_ub)
                                    if with_residual:
                                        clear_padding(residual_ub)
                                    if out_config.with_sf and not reuse_input:
                                        clear_padding(out_ub)
                                rstd_base = row_base * block_g if with_rstd else 0
                                for row in T.serial(T.uint16(block_m)):
                                    for group in T.serial(T.uint16(block_g)):
                                        square_sum = reduce_group(x_ub, residual_ub, row, group * rms_group_size)
                                        S.vsts(rstd_ub[rstd_base + row * block_g + group], square_sum, dist='ONEPT_B32')
                                S.mem_bar('VST_VLD')
                                offset = rstd_base % vec_size
                                for vec in T.serial(T.uint16(num_rstd_vecs)):
                                    aligned_index = rstd_base - offset + vec * vec_size
                                    rstd_values = compute_rstd(S.vld(rstd_ub[aligned_index]))
                                    if with_rstd and rstds_per_tile % vec_size != 0:
                                        # Replace this tile's square sums in place, preserving earlier tiles' RSTD.
                                        indices = T.reinterpret(S.vci(vec * vec_size - offset, T.int32), 'uint32x64')
                                        mask = S.vcmps(indices, rstds_per_tile, op='lt')
                                        S.vsts(rstd_ub[aligned_index], rstd_values, mask)
                                    else:
                                        S.vsts(rstd_ub[aligned_index], rstd_values)
                                S.mem_bar('VST_VLD')
                                apply_tile(residual_ub if with_residual else x_ub, weight_ub, out_ub, rstd_ub, rstd_base)
                            if out_config.with_sf:
                                quantize_tile(
                                    out_ub,
                                    quantized_ub,
                                    out_bf16_ub,
                                    sf_scratch_ub,
                                    scales_ub,
                                    row_base if transpose_sf else 0,
                                )
                            T.copy((quantized_ub if out_config.with_sf else out_ub)[:, :block_k], out[batch_id, token_start, col_start])
                            if out_config.with_sf and not transpose_sf:
                                T.copy(scales_ub[:, :num_groups], out_sf[batch_id, token_start, col_start // num_per_channels])
                            if with_out_bf16:
                                T.copy(out_bf16_ub[:, :block_k], out_bf16[batch_id, token_start, col_start])
                            if with_residual:
                                T.copy(residual_ub[:, :block_k], residual_out[batch_id, token_start, col_start])
                    if transpose_sf:
                        with T.SimdVF():
                            indices = S.vld(transpose_indices_ub[0])
                            for vector in T.unroll(T.ceildiv(num_sf_words * sf_rows, sf_lanes), explicit=True):
                                pack = vector * (sf_lanes // sf_rows)
                                values = S.vgather2(scale_words[0, pack], indices)
                                S.vsts(transposed_sf_ub[pack, 0], values, dist=sf_store_dist)
                        T.copy(
                            transposed_sf_ub[:num_sf_words, :rows_per_block],
                            out_sf_words[batch_id, col_start // num_per_channels // pack_factor, token_base],
                        )
                    if with_rstd:
                        T.copy(rstd_rows[:rows_per_block, :], rstd_groups[batch_id, token_base, group_block * block_g])

    return norm_forward_kernel

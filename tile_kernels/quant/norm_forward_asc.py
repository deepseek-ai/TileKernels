import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from tile_kernels.config import get_max_ub_per_vector_core
from tile_kernels.quant.common import CastOutputConfig, get_sf_shape
from tile_kernels.quant.rstd import make_rstd_impl
from tile_kernels.utils import align, ceil_div


@T.macro
def load_f32(src: T.Ref):
    """Load 64 contiguous elements as FP32."""
    if src.dtype == T.bfloat16:
        return S.vcvt(S.vld(src, dist='UNPK_B16'), T.float32, part=0)
    return S.vld(src)


@T.macro
def load_f32_pair(src0: T.Ref, src1: T.Ref):
    """Load BF16 even/odd lanes or FP32 contiguous halves as two FP32 vectors."""
    if src0.dtype == T.bfloat16:
        packed = S.vld(src0)
        return S.vcvt(packed, T.float32, part=0), S.vcvt(packed, T.float32, part=1)
    return S.vld(src0), S.vld(src1)


@T.macro
def store_pair(dst0: T.Ref, dst1: T.Ref, x0, x1):
    """Store a pair in the same lane order as load_f32_pair."""
    if dst0.dtype == T.bfloat16:
        low = S.vcvt(x0, T.bfloat16, part=0)
        high = S.vcvt(x1, T.bfloat16, part=1)
        packed = S.vor(T.reinterpret(low, 'uint16x128'), T.reinterpret(high, 'uint16x128'))
        S.vsts(dst0, T.reinterpret(packed, 'bfloat16x128'), dist='NORM_B16')
    else:
        S.vsts(dst0, x0)
        S.vsts(dst1, x1)


@T.macro
def get_sf_and_inv_simd(amax, out_config: CastOutputConfig):
    """SIMD counterpart of get_sf_and_inv; packed SF is returned in uint32 lanes."""
    clamped = S.vmaxs(amax, out_config.clamp_min_value)
    max_value = float(T.max_value(out_config.dtype))
    if not out_config.round_sf:
        maximum = S.vdup(max_value, T.float32)
        return S.vdiv(clamped, maximum), S.vdiv(maximum, clamped)
    bits = T.reinterpret(S.vmuls(clamped, 1.0 / max_value), 'uint32x64')
    exponent = S.vadds(S.vshrs(S.vsub(bits, S.vdup(1, T.uint32)), 23), 1)
    inverse = T.reinterpret(S.vshls(S.vsub(S.vdup(254, T.uint32), exponent), 23), 'float32x64')
    if out_config.use_packed_ue8m0:
        return exponent, inverse
    return T.reinterpret(S.vshls(exponent, 23), 'float32x64'), inverse


@tilelang.jit
def get_norm_forward_kernel_asc(
    hidden: int,
    with_weight: bool,
    with_residual: bool,
    eps: float,
    out_scale: float,
    x_dtype: T.dtype,
    out_config: CastOutputConfig,
    num_vec_cores: int,
    with_out_bf16: bool = False,
    with_rstd: bool = False,
):
    assert hidden > 0
    assert x_dtype in (T.bfloat16, T.float32)
    if out_config.with_sf:
        assert out_config.sf_block == (1, 32)
    padded_hidden = align(hidden, 256)
    row_bytes = padded_hidden * x_dtype.bytes
    vec_size = 64
    load_size = 256 // x_dtype.bytes
    num_per_channels = out_config.sf_block[1]
    use_bf16_quant = x_dtype == T.bfloat16 and out_config.round_sf
    num_parts = load_size // vec_size
    num_accumulators = min(4, padded_hidden // vec_size)
    reduction_cols = num_accumulators * vec_size

    # SF layout only affects stores. All formats share normalization and quantization.
    transpose_sf = out_config.with_sf and out_config.use_tma_aligned_col_major_sf
    num_groups = hidden // num_per_channels
    sf_size = align(num_groups, vec_size)
    # Transpose SF as raw bits: pairs of exponent bytes or individual FP32 scales.
    sf_word_dtype = T.uint16 if out_config.use_packed_ue8m0 else T.uint32
    sf_pack = sf_word_dtype.bytes // out_config.sf_dtype.bytes
    sf_lanes = 256 // sf_word_dtype.bytes
    sf_store_dist = f'NORM_B{sf_word_dtype.bits}'
    num_sf_words = ceil_div(num_groups, sf_pack)

    # UB capacity is static; the number of tiles grouped for stores is chosen at runtime.
    tile_capacity = max(1, 32768 // (1 + with_residual) // row_bytes)
    min_rows_per_block = 8 if transpose_sf else 4
    max_rows_per_block = 16 if transpose_sf else 128
    rows_per_tile = min(max_rows_per_block, vec_size, 1 << (tile_capacity.bit_length() - 1))
    sf_rows = max(max_rows_per_block, 32 // sf_word_dtype.bytes) if transpose_sf else rows_per_tile
    rstd_out_size = align(max_rows_per_block, vec_size) if with_rstd else vec_size

    tile_bytes = rows_per_tile * row_bytes
    stage_bytes = (
        (2 + with_residual) * tile_bytes  # x_ub, out_ub, residual_ub
        + (rows_per_tile * padded_hidden * out_config.dtype.bytes if out_config.with_sf else 0)  # quantized_ub
        + (rows_per_tile * sf_size * out_config.sf_dtype.bytes if out_config.with_sf and not transpose_sf else 0)  # scales_ub
        + (rows_per_tile * padded_hidden * T.bfloat16.bytes if with_out_bf16 and x_dtype != T.bfloat16 else 0)  # out_bf16_ub
    )
    fixed_bytes = (
        (row_bytes if with_weight else 0)  # weight_ub
        + rstd_out_size * T.float32.bytes  # rstd_out_ub
        + (rows_per_tile * sf_size * T.float32.bytes if out_config.with_sf else 0)  # sf_scratch_ub
        + (sf_rows * sf_size * out_config.sf_dtype.bytes if transpose_sf else 0)  # scales_ub
        + (sf_size // sf_pack * sf_rows * sf_word_dtype.bytes if transpose_sf else 0)  # transposed_sf_ub
        + (sf_lanes * sf_word_dtype.bytes if transpose_sf else 0)  # transpose_indices_ub
    )
    available_bytes = get_max_ub_per_vector_core(use_simt=False) - fixed_bytes
    num_stages = min(4, available_bytes // stage_bytes)
    # Reuse the input buffer for the output when it buys a deeper pipeline.
    reused_stages = min(4, available_bytes // (stage_bytes - tile_bytes))
    reuse_input = num_stages < 3 and reused_stages > num_stages
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
        # Clear whole tail vectors before DMA overwrites their valid elements.
        zero = S.vdup(0.0, buffer.dtype)
        for vector in T.serial(T.uint16(hidden // load_size), T.uint16(padded_hidden // load_size)):
            for row in T.serial(T.uint16(rows_per_tile)):
                S.vsts(buffer[row, vector * load_size], zero)

    compute_rstd = make_rstd_impl(hidden, eps)

    @T.macro
    def reduce_row(x_ub, residual_ub, row):
        sums = S.alloc_local((num_accumulators,), T.float32)
        for i in T.unroll(num_accumulators, explicit=True):
            sums[i] = S.vdup(0.0, T.float32)
        for group in T.serial(T.uint16(T.ceildiv(padded_hidden, reduction_cols))):
            for chunk in T.unroll(reduction_cols // load_size, explicit=True):
                col = group * reduction_cols + chunk * load_size
                if col < padded_hidden:
                    x = S.alloc_var(x_dtype)
                    x = S.vld(x_ub[row, col])
                    if with_residual:
                        x = S.vadd(x, S.vld(residual_ub[row, col]))
                        S.vsts(residual_ub[row, col], x)
                    low = S.vcvt(x, T.float32, part=0) if x_dtype == T.bfloat16 else x
                    S.vmula(sums[chunk * num_parts], low, low)
                    if x_ub.dtype == T.bfloat16:
                        high = S.vcvt(x, T.float32, part=1)
                        S.vmula(sums[chunk * num_parts + 1], high, high)
        for i in T.unroll(1, num_accumulators, explicit=True):
            sums[0] = S.vadd(sums[0], sums[i])
        return S.vcadd(sums[0])

    @T.macro
    def apply_tile(x_ub, weight_ub, out_ub, rstd_values):
        for pair in T.serial(T.uint16(padded_hidden // 128)):
            col = pair * 128
            if with_weight:
                w0, w1 = load_f32_pair(weight_ub[col], weight_ub[col + 64])
                if out_scale != 1.0:
                    w0, w1 = S.vmuls(w0, out_scale), S.vmuls(w1, out_scale)
            for row in T.serial(T.uint16(rows_per_tile)):
                inv_rms = S.vselr(rstd_values, S.vdup(row, T.uint32))
                x0, x1 = load_f32_pair(x_ub[row, col], x_ub[row, col + 64])
                y0, y1 = S.vmul(x0, inv_rms), S.vmul(x1, inv_rms)
                if with_weight:
                    y0, y1 = S.vmul(y0, w0), S.vmul(y1, w1)
                elif out_scale != 1.0:
                    y0, y1 = S.vmuls(y0, out_scale), S.vmuls(y1, out_scale)
                store_pair(out_ub[row, col], out_ub[row, col + 64], y0, y1)

    @T.macro
    def quantize_tile(source, quantized, auxiliary, scratch, scales, scale_row_base):
        inverse_scales = T.Tensor((rows_per_tile, sf_size), T.bfloat16 if use_bf16_quant else T.float32, scratch.data)
        with T.SimdVF():
            low_mask = S.pset(T.float32.bits, 'PAT_VL32')
            store_mask = S.pset(32, 'PAT_VL8')
            for row in T.serial(T.uint16(rows_per_tile)):
                for tile in T.serial(T.uint16(padded_hidden // 256)):
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
            for row in T.serial(T.uint16(rows_per_tile)):
                for group_tile in T.serial(T.uint16(sf_size // vec_size)):
                    group = group_tile * vec_size
                    sf, inv = get_sf_and_inv_simd(S.vld(scratch[row, group]), out_config)
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
                for row in T.serial(T.uint16(rows_per_tile)):
                    for tile in T.serial(T.uint16(padded_hidden // 256)):
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
                for row in T.serial(T.uint16(rows_per_tile)):
                    for vector in T.serial(T.uint16(padded_hidden // vec_size)):
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
        weight: T.Tensor[(hidden,), x_dtype],
        rstd: T.Tensor[(batch, num_tokens), T.float32],
        residual: T.StridedTensor[(batch, num_tokens, hidden), (residual_batch_stride, residual_stride, 1), x_dtype],
        residual_out: T.StridedTensor[(batch, num_tokens, hidden), (residual_out_batch_stride, residual_out_stride, 1), x_dtype],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            x_ub = T.alloc_shared((rows_per_tile, padded_hidden), x_dtype)
            out_ub = x_ub if reuse_input else T.alloc_shared((rows_per_tile, padded_hidden), x_dtype)
            weight_ub = T.alloc_shared((padded_hidden,), x_dtype)
            residual_ub = T.alloc_shared((rows_per_tile, padded_hidden), x_dtype)
            out_bf16_ub = out_ub if x_dtype == T.bfloat16 else T.alloc_shared((rows_per_tile, padded_hidden), T.bfloat16)
            rstd_out_ub = T.alloc_shared((rstd_out_size,), T.float32)
            quantized_ub = T.alloc_shared((rows_per_tile, padded_hidden), out_config.dtype)
            sf_scratch_ub = T.alloc_shared((rows_per_tile, sf_size), T.float32)
            scales_ub = T.alloc_shared((sf_rows, sf_size), out_config.sf_dtype)
            if transpose_sf:
                transposed_sf_ub = T.alloc_shared((sf_size // sf_pack, sf_rows), sf_word_dtype)
                transpose_indices_ub = T.alloc_shared((sf_lanes,), sf_word_dtype)
                scale_words = T.Tensor((sf_rows, sf_size // sf_pack), sf_word_dtype, scales_ub.data)
                out_sf_words = T.StridedTensor(
                    (batch, num_sf_words, num_tokens),
                    (out_sf_batch_stride // sf_pack, out_sf_stride // sf_pack, 1),
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
                    rstd_out_ub: 1,
                    sf_scratch_ub: 1,
                }
            )
            if transpose_sf:
                with T.SimdVF():
                    lane = T.reinterpret(S.vci(0, f'int{sf_word_dtype.bits}'), f'{sf_word_dtype}x{sf_lanes}')
                    row = S.vand(lane, S.vdup(sf_rows - 1, sf_word_dtype))
                    pack = S.vshrs(lane, sf_rows.bit_length() - 1)
                    S.vsts(transpose_indices_ub[0], S.vadd(S.vmuls(row, sf_size // sf_pack), pack), dist=sf_store_dist)
            if with_weight:
                T.copy(weight[0], weight_ub)
            rows_per_block = T.alloc_var(
                T.int32, init=T.clamp(T.ceildiv(num_tokens, num_vec_cores * rows_per_tile) * rows_per_tile, min_rows_per_block, max_rows_per_block)
            )
            num_token_blocks = T.ceildiv(num_tokens, rows_per_block)
            for batch_id, pid_token in T.Persistent([batch, num_token_blocks], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
                token_base = pid_token * rows_per_block
                for tile_index in T.serial(rows_per_block // rows_per_tile):
                    row_base = tile_index * rows_per_tile
                    token_start = token_base + row_base
                    if token_start < num_tokens:
                        if hidden != padded_hidden:
                            with T.SimdVF():
                                clear_padding(x_ub)
                                if with_residual:
                                    clear_padding(residual_ub)
                        T.copy(x[batch_id, token_start, 0], x_ub, l2_cache_ctrl='NOTALLOC_KEEP')
                        if with_residual:
                            T.copy(residual[batch_id, token_start, 0], residual_ub, l2_cache_ctrl='NOTALLOC_KEEP')
                        with T.SimdVF():
                            rstd_base = row_base if with_rstd else 0
                            for row in T.serial(T.uint16(rows_per_tile)):
                                square_sum = reduce_row(x_ub, residual_ub, row)
                                S.vsts(rstd_out_ub[rstd_base + row], square_sum, dist='ONEPT_B32')
                            S.mem_bar('VST_VLD')
                            offset = rstd_base % vec_size
                            rstd_values = compute_rstd(S.vld(rstd_out_ub[rstd_base - offset]))

                            if with_rstd:
                                # Replace this tile's square sums in place, preserving earlier tiles' RSTD.
                                indices = S.vand(S.vci(-offset, T.int32), S.vdup(vec_size - 1, T.int32))
                                mask = S.vcmps(indices, rows_per_tile, op='lt')
                                S.vsts(rstd_out_ub[rstd_base - offset], rstd_values, mask)
                                indices = S.vand(S.vci(offset, T.int32), S.vdup(vec_size - 1, T.int32))
                                rstd_values = S.vselr(rstd_values, indices)
                            apply_tile(residual_ub if with_residual else x_ub, weight_ub, out_ub, rstd_values)
                        if out_config.with_sf:
                            quantize_tile(
                                out_ub,
                                quantized_ub,
                                out_bf16_ub,
                                sf_scratch_ub,
                                scales_ub,
                                row_base if transpose_sf else 0,
                            )
                        T.copy(quantized_ub if out_config.with_sf else out_ub, out[batch_id, token_start, 0])
                        if out_config.with_sf and not transpose_sf:
                            T.copy(scales_ub[:, :num_groups], out_sf[batch_id, token_start : token_start + rows_per_tile, :num_groups])
                        if with_out_bf16:
                            T.copy(out_bf16_ub, out_bf16[batch_id, token_start, 0])
                        if with_residual:
                            T.copy(residual_ub, residual_out[batch_id, token_start, 0])
                if transpose_sf:
                    with T.SimdVF():
                        indices = S.vld(transpose_indices_ub[0])
                        for vector in T.unroll(T.ceildiv(num_sf_words * sf_rows, sf_lanes), explicit=True):
                            pack = vector * (sf_lanes // sf_rows)
                            values = S.vgather2(scale_words[0, pack], indices)
                            S.vsts(transposed_sf_ub[pack, 0], values, dist=sf_store_dist)
                    T.copy(transposed_sf_ub[:num_sf_words, :rows_per_block], out_sf_words[batch_id, 0, token_base])
                if with_rstd:
                    T.copy(rstd_out_ub[:rows_per_block], rstd[batch_id, token_base])

    return norm_forward_kernel

import math

import tilelang

from tile_kernels.quant.common import *

# Re-bind below the star import: tile_kernels.quant.common exports its own `T`.
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S


@tilelang.jit
def get_per_channel_cast_kernel_asc(
    hidden: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
    num_vec_cores: int,
):
    num_per_tokens = out_config.sf_block[0]
    assert num_per_tokens == 32, 'Ascend cast supports num_per_tokens == 32 only'
    if in_config.with_sf:
        num_per_channels = in_config.sf_block[1]
        assert num_per_channels == 32, 'Ascend rescale supports num_per_channels == 32 only'
    else:
        assert in_config.dtype in (T.bfloat16, T.float32), 'per_channel_cast supports bf16, fp32 or e4m3 (rescale) input only'

    quant_max = 448.0
    assert out_config.clamp_min_value >= quant_max * (2 ** (-126)), 'Ascend scale exponent fast path requires clamp to normal number'

    use_bf16_compute = in_config.dtype == T.bfloat16 or (in_config.with_sf and in_config.use_packed_ue8m0)
    # Packed input with rounded output decode at 1024x256, the other bf16 paths pick 256x128.
    use_large_block = in_config.with_sf and in_config.use_packed_ue8m0 and out_config.round_sf
    vector_width = 256  # SIMD vector width in bytes
    vector_alignment = 32  # Minimum byte alignment for vector loads
    max_block_k = 1024 if use_large_block else 256
    block_k = math.gcd(max_block_k, hidden)
    block_m = min(128, (16384 if in_config.dtype == T.float32 else 32768) // block_k)
    sf_rows_per_group = block_m // num_per_tokens

    assert block_m % num_per_tokens == 0, (
        f'Ascend per channel cast requires block_m % num_per_tokens == 0, got block_m={block_m}, num_per_tokens={num_per_tokens}'
    )
    assert block_k % 128 == 0, f'Ascend per channel cast requires block_k % 128 == 0, got block_k={block_k} (hidden={hidden})'
    num_k_blocks = hidden // block_k

    pack_factor = get_packed_ue8m0_pack_factor()
    assert pack_factor == 2
    if in_config.with_sf:
        sf_in_block_k = num_per_channels * pack_factor if in_config.use_packed_ue8m0 else num_per_channels
    num_blocks_per_group = pack_factor if out_config.use_packed_ue8m0 else 1

    num_stages = 2

    num_tokens = T.dynamic('num_tokens')
    sf_stride = T.dynamic('sf_stride')
    num_groups = T.ceildiv(num_tokens, block_m * num_blocks_per_group) * num_k_blocks
    num_waves = T.ceildiv(num_groups, num_vec_cores) * num_blocks_per_group
    sf_out_shape = get_sf_shape((num_tokens, hidden), out_config)
    x_sf_shape = get_sf_shape((num_tokens, hidden), in_config)
    packed_col_major_sf = in_config.use_tma_aligned_col_major_sf and in_config.use_packed_ue8m0
    if packed_col_major_sf:
        x_sf_shape = (x_sf_shape[0], x_sf_shape[1] // pack_factor)
    in_sf_stride = T.dynamic('in_sf_stride') if in_config.with_sf else x_sf_shape[1]
    sf_in_dtype = T.int16 if packed_col_major_sf else in_config.sf_dtype
    quant_chunk_size = 128 if use_bf16_compute and (not use_large_block or block_k == 512) else min(256, block_k)
    # Count 64-lane fp32 vectors; each pair combines into one 128-lane bf16 vector.
    num_vecs = quant_chunk_size // 64
    compute_vec_factor = 2 if use_bf16_compute else 1

    if in_config.with_sf:
        if in_config.use_tma_aligned_col_major_sf:
            # Col-major layout is read via per-lane gather, not wide loads, so no tail padding needed.
            sf_in_ub_shape = (block_k // sf_in_block_k, block_m)
        else:
            # Row stride stays vector_alignment-byte aligned for the aligned vector loads; vector_alignment // bytes = element floor.
            sf_in_cols = max(block_k // num_per_channels, vector_alignment // sf_in_dtype.bytes)
            # Pad the tail: a full vector_width (256B) load spills past the last row.
            sf_load_elements = vector_width // sf_in_dtype.bytes
            sf_in_rows = block_m + T.ceildiv(sf_load_elements, sf_in_cols) - 1
            sf_in_ub_shape = (sf_in_rows, sf_in_cols)
        dequant_ub_dtype = T.bfloat16 if use_bf16_compute else T.float32

    @T.macro
    def quantize_block(x_ub, out_ub, sf_out_ub, sf_in_ub, dequant_ub, sf_row_base=0):
        with T.SimdVF():
            # Prepare and build gather offsets per input sf layout
            sf_dist = 'PK4_B32' if out_config.use_packed_ue8m0 else 'NORM_B32'
            amax = S.alloc_local((num_vecs,), T.float32)
            sf_inv = S.alloc_local((num_vecs,), T.float32)

            if use_bf16_compute:
                zero_bf16 = S.vdup(0.0, T.bfloat16)
                bf16_abs_mask = T.reinterpret(S.vdup(0x7FFF, T.uint16), 'bfloat16x128')
                if in_config.with_sf:
                    sf_lane = T.reinterpret(S.vshrs(S.vci(0, T.int16), 5), 'uint16x128')  # 0x32 1x32 2x32 3x32
                    if in_config.use_tma_aligned_col_major_sf:
                        # Col-major: int16 entries viewed as byte pairs, flat (k-block, token-pair) order.
                        sf_flat_ub = T.view(sf_in_ub, (block_k // sf_in_block_k * block_m * pack_factor,), T.uint8)
                        sf_offsets = S.vadd(
                            S.vmuls(S.vshrs(sf_lane, 1), block_m * pack_factor),
                            S.vand(sf_lane, S.vdup(1, T.uint16)),
                        )  # 0x32 1x32 (2 * block_m)x32 (2 * block_m + 1)x32
                        sf_mask = S.pset(8)
                        sf_col_stride = block_m * pack_factor
                        sf_row_stride = pack_factor
                    else:
                        sf_flat_ub = T.view(sf_in_ub, (sf_in_rows * sf_in_cols,), T.uint8)
                        sf_offsets = sf_lane
                        sf_mask = None
                        sf_col_stride = 1
                        sf_row_stride = sf_in_cols
            else:
                f32_abs_mask = T.reinterpret(S.vdup(0x7FFFFFFF, T.uint32), 'float32x64')
                if in_config.with_sf:
                    sf_lane = T.reinterpret(S.vshrs(S.vci(0, T.int32), 5), 'uint32x64')  # 0x32 1x32

            for sf_row in T.serial(sf_rows_per_group):
                row_base = sf_row * num_per_tokens
                # Dequantize and accumulate amax
                for chunk in T.serial(block_k // quant_chunk_size):
                    chunk_base = chunk * quant_chunk_size
                    if use_bf16_compute:
                        amax_bf16 = S.alloc_local((num_vecs // compute_vec_factor,), T.bfloat16)
                    for vec in T.unroll(num_vecs // compute_vec_factor, explicit=True):
                        vec_col = vec * 64 * compute_vec_factor
                        if use_bf16_compute:
                            # bf16 path
                            amax_bf16[vec] = S.vdup(0.0, T.bfloat16)
                            if in_config.with_sf:
                                if in_config.use_tma_aligned_col_major_sf:
                                    sf_col = chunk * (quant_chunk_size // 64) + vec * 2
                                else:
                                    sf_col = chunk * (quant_chunk_size // 32) + vec * 4
                            for row in T.serial(num_per_tokens):
                                if in_config.with_sf:
                                    raw_f32_0 = S.vcvt(S.vld(x_ub[row_base + row, chunk_base + vec_col], dist='UNPK4_B8'), T.float32)
                                    raw_f32_1 = S.vcvt(S.vld(x_ub[row_base + row, chunk_base + vec_col + 64], dist='UNPK4_B8'), T.float32)
                                    _, raw_bf16 = S.vdintlv(T.reinterpret(raw_f32_0, 'bfloat16x128'), T.reinterpret(raw_f32_1, 'bfloat16x128'))
                                    sf_exponents = S.vgather2(
                                        sf_flat_ub[(row_base + row) * sf_row_stride + sf_col * sf_col_stride],
                                        sf_offsets,
                                        mask=sf_mask,
                                    )
                                    sf_bits = S.vshls(sf_exponents, 7)
                                    values_bf16 = S.vmul(raw_bf16, T.reinterpret(sf_bits, 'bfloat16x128'))
                                    S.vsts(dequant_ub[row, vec_col], values_bf16, dist='NORM_B16')
                                else:
                                    values_bf16 = S.vld(x_ub[row_base + row, chunk_base + vec_col])
                                amax_bf16[vec] = S.vmax(amax_bf16[vec], S.vand(values_bf16, bf16_abs_mask))
                        else:
                            # f32 path
                            amax[vec] = S.vdup(0.0, T.float32)
                            if in_config.with_sf:
                                if in_config.use_tma_aligned_col_major_sf:
                                    sf_offsets = S.vadds(S.vmuls(sf_lane, block_m), (chunk * 8 + vec * 2) * block_m)
                                else:
                                    sf_offsets = S.vadds(sf_lane, vec * 2)
                            for row in T.serial(num_per_tokens):
                                if in_config.with_sf:
                                    if in_config.use_tma_aligned_col_major_sf:
                                        input_scale = S.vgather2(sf_in_ub[0, 0], S.vadds(sf_offsets, row_base + row))
                                    else:
                                        compact_sf = S.vld(sf_in_ub[row_base + row, chunk * 8])
                                        input_scale = S.vselr(compact_sf, sf_offsets)
                                    values_fp32 = S.vmul(
                                        S.vcvt(S.vld(x_ub[row_base + row, chunk_base + vec_col], dist='UNPK4_B8'), T.float32),
                                        input_scale,
                                    )
                                    S.vsts(dequant_ub[row, vec_col], values_fp32)
                                else:
                                    values_fp32 = S.vld(x_ub[row_base + row, chunk_base + vec_col])
                                amax[vec] = S.vmax(amax[vec], S.vand(values_fp32, f32_abs_mask))
                        if use_bf16_compute:
                            # Merge the 128-lane bf16 amax into two f32 half-vectors
                            amax_lo, amax_hi = S.vintlv(zero_bf16, amax_bf16[vec])
                            amax[vec * 2] = T.reinterpret(amax_lo, 'float32x64')
                            amax[vec * 2 + 1] = T.reinterpret(amax_hi, 'float32x64')

                    # Mem bar on dequant_ub
                    S.mem_bar('VST_VLD')
                    for pair in T.unroll(num_vecs // 2, explicit=True):
                        # Write sf_out_ub
                        for half in T.unroll(2, explicit=True):
                            vec = pair * 2 + half
                            sf, sf_inv[vec] = get_sf_and_inv_asc(amax[vec], out_config)
                            S.vsts(sf_out_ub[sf_row_base + sf_row, chunk_base + vec * 64], sf, dist=sf_dist)

                        # Calculate and write out_ub
                        for half in T.unroll(2, explicit=True):
                            vec = pair * 2 + half
                            vec_col = vec * 64
                            for row in T.serial(num_per_tokens):
                                if use_bf16_compute:
                                    value_ub, value_row, value_col = (dequant_ub, row, 0) if in_config.with_sf else (x_ub, row_base + row, chunk_base)
                                    values_fp32 = S.vcvt(S.vld(value_ub[value_row, value_col + vec_col], dist='UNPK_B16'), T.float32)
                                elif in_config.with_sf:
                                    values_fp32 = S.vld(dequant_ub[row, vec_col])
                                else:
                                    values_fp32 = S.vld(x_ub[row_base + row, chunk_base + vec_col])
                                scaled = S.vmul(values_fp32, sf_inv[vec])
                                S.vsts(
                                    out_ub[row_base + row, chunk_base + vec_col],
                                    S.vcvt(scaled, T.float8_e4m3fn),
                                    dist='PK4_B32',
                                )

    @T.macro
    def pack_sf_rows(unpacked_sf_ub, packed_sf_ub, unpacked_row, packed_row):
        with T.SimdVF():
            for col in T.serial(0, block_k, vector_width):
                packed_0, packed_1 = S.vintlv(S.vld(unpacked_sf_ub[unpacked_row, col]), S.vld(unpacked_sf_ub[unpacked_row + 1, col]))
                S.vsts(packed_sf_ub[packed_row, col * 2], packed_0, dist='NORM_B8')
                if block_k - col >= vector_width:
                    S.vsts(packed_sf_ub[packed_row, col * 2 + vector_width], packed_1, dist='NORM_B8')

    @T.prim_func
    def per_channel_cast_kernel(
        x: T.Tensor[(num_tokens, hidden), in_config.dtype],
        out: T.Tensor[(num_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[sf_out_shape, (sf_stride, 1), out_config.sf_dtype],
        x_sf_invs: T.StridedTensor[x_sf_shape, (in_sf_stride, 1), sf_in_dtype],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            x_ub = T.alloc_shared((block_m, block_k), in_config.dtype)
            out_ub = T.alloc_shared((block_m, block_k), out_config.dtype)
            versions = {x_ub: num_stages, out_ub: num_stages}

            if in_config.with_sf:
                sf_in_ub = T.alloc_shared(sf_in_ub_shape, sf_in_dtype)
                dequant_ub = T.alloc_shared((num_per_tokens, quant_chunk_size), dequant_ub_dtype)
                versions[sf_in_ub] = num_stages

            sf_out_ub_dtype = T.uint8 if out_config.use_packed_ue8m0 else T.float32
            aligned_block_k = align(block_k, vector_width // sf_out_ub_dtype.bytes)
            sf_out_ub = T.alloc_shared((sf_rows_per_group * num_blocks_per_group, aligned_block_k), sf_out_ub_dtype)
            versions[sf_out_ub] = 1 if out_config.use_packed_ue8m0 else num_stages
            if out_config.use_packed_ue8m0:
                packed_sf_ub = T.alloc_shared((sf_rows_per_group, block_k * pack_factor), T.uint8)
                versions[packed_sf_ub] = num_stages
            T.annotate_buffer_versions(versions)

            for wave_id in T.Persistent(
                [num_waves],
                1,
                0,
                group_size=1,
                num_stages=num_stages,
                annotations={'multi_buffer_eligible': [buf for buf, n in versions.items() if n != 1]},
            ):
                # Map `wave_id` into (`group_id`, `block_id_in_group`)
                group_id = (wave_id // num_blocks_per_group) * num_vec_cores + core_id
                block_id_in_group = wave_id % num_blocks_per_group

                if group_id < num_groups:
                    # Map (`group_id`, `block_id_in_group`) to (`m_block_id`, `k_block_id`)
                    m_group_id = group_id // num_k_blocks
                    k_block_id = group_id % num_k_blocks
                    m_block_id = m_group_id * num_blocks_per_group + block_id_in_group

                    col = k_block_id * block_k
                    if num_blocks_per_group == 1 or m_block_id * block_m < num_tokens:
                        T.copy(x[m_block_id * block_m, col], x_ub, l2_cache_ctrl='NOTALLOC_KEEP')
                        if in_config.with_sf:
                            if in_config.use_tma_aligned_col_major_sf:
                                T.copy(
                                    x_sf_invs[col // sf_in_block_k, m_block_id * block_m],
                                    sf_in_ub[: block_k // sf_in_block_k, :block_m],
                                )
                            else:
                                T.copy(
                                    x_sf_invs[m_block_id * block_m, col // num_per_channels],
                                    sf_in_ub[:block_m, : block_k // num_per_channels],
                                )
                        quantize_block(
                            x_ub,
                            out_ub,
                            sf_out_ub,
                            sf_in_ub if in_config.with_sf else None,
                            dequant_ub if in_config.with_sf else None,
                            sf_row_base=block_id_in_group * sf_rows_per_group,
                        )
                        T.copy(out_ub, out[m_block_id * block_m, col])

                    if block_id_in_group == num_blocks_per_group - 1:
                        sf_row = m_group_id * sf_rows_per_group
                        # Pack two stage output sf together if using ue8m0 scale factor output
                        if out_config.use_packed_ue8m0:
                            for packed_row in T.serial(sf_rows_per_group):
                                if m_group_id * (block_m * num_blocks_per_group) + 2 * packed_row * num_per_tokens < num_tokens:
                                    pack_sf_rows(sf_out_ub, packed_sf_ub, 2 * packed_row, packed_row)
                            valid_rows = T.alloc_var(T.int32, init=T.min(sf_rows_per_group, sf_out_shape[0] - sf_row))
                            T.copy(packed_sf_ub[0:valid_rows, :], out_sf[sf_row, col * pack_factor])
                        else:
                            T.copy(sf_out_ub, out_sf[sf_row, col])

    return per_channel_cast_kernel

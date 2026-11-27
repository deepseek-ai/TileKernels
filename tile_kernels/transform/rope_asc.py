from typing import Any

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from tile_kernels.config import get_max_ub_per_vector_core

SIMD_LANES = 64


@tilelang.jit
def get_rope_kernel_asc(
    dtype: T.dtype,
    positions_dtype: T.dtype,
    nheads: int,
    rotary_dim: int,
    has_position_tensor: bool,
    interleaved: bool,
    conjugate: bool,
    cos_sin_stride: int,
    x_stride_0: int,
    x_stride_1: int,
    num_vec_cores: int,
) -> Any:
    assert dtype in (T.bfloat16, T.float32)
    assert positions_dtype in (T.int32, T.int64)
    assert rotary_dim in (64, 128)

    half_dim = rotary_dim // 2
    dtype_size = dtype.bytes
    bytes_per_row = nheads * rotary_dim * dtype_size + rotary_dim * 4
    max_ub_size = get_max_ub_per_vector_core(use_simt=False)
    num_stages = min(3, max_ub_size // bytes_per_row)
    ub_block_tokens = min(SIMD_LANES, max_ub_size // (num_stages * bytes_per_row))
    block_tokens = 1 << (ub_block_tokens.bit_length() - 1)
    can_bulk_copy = x_stride_0 == nheads * x_stride_1 or x_stride_1 == rotary_dim

    batch = T.dynamic('batch')
    seqlen = T.dynamic('seqlen')
    seqlen_ro = T.dynamic('seqlen_ro')
    x_stride_b = T.dynamic('x_stride_b')
    positions_stride_b = T.dynamic('positions_stride_b')

    @T.macro
    def rotate(
        first: T.float32x64,
        second: T.float32x64,
        cosine: T.float32x64,
        sine: T.float32x64,
        neg_sine: T.float32x64,
    ) -> tuple[T.float32x64, T.float32x64]:
        out_first, out_second = S.alloc_var(T.float32), S.alloc_var(T.float32)
        out_first = S.vmul(second, sine)
        S.vmula(out_first, first, cosine)
        out_second = S.vmul(second, cosine)
        S.vmula(out_second, first, neg_sine)
        return out_first, out_second

    @T.macro
    def load2(
        buffer: T.Buffer,
        row: T.int32,
        first_head: T.int32,
        first_column: T.int32,
        second_head: T.int32,
        second_column: T.int32,
        packed_bf16: bool,
    ) -> tuple[T.float32x64, T.float32x64]:
        first, second = S.alloc_var(T.float32), S.alloc_var(T.float32)
        if dtype == T.bfloat16:
            if packed_bf16:
                packed = S.vld(buffer[row, first_head, first_column])
                first = S.vcvt(packed, T.float32, part=0)
                second = S.vcvt(packed, T.float32, part=1)
            else:
                first = S.vcvt(S.vld(buffer[row, first_head, first_column], dist='UNPK_B16'), T.float32, part=0)
                second = S.vcvt(S.vld(buffer[row, second_head, second_column], dist='UNPK_B16'), T.float32, part=0)
        else:
            first = S.vld(buffer[row, first_head, first_column])
            second = S.vld(buffer[row, second_head, second_column])
        return first, second

    @T.macro
    def store2(
        buffer: T.Buffer,
        row: T.int32,
        first_head: T.int32,
        first_column: T.int32,
        second_head: T.int32,
        second_column: T.int32,
        first: T.float32x64,
        second: T.float32x64,
        has_second: bool,
        packed_bf16: bool,
    ) -> None:
        if dtype == T.bfloat16:
            if packed_bf16:
                first_bf16 = T.reinterpret(S.vcvt(first, T.bfloat16, part=0), 'uint16x128')
                second_bf16 = T.reinterpret(S.vcvt(second, T.bfloat16, part=1), 'uint16x128')
                packed = T.reinterpret(S.vor(first_bf16, second_bf16), 'bfloat16x128')
                S.vsts(buffer[row, first_head, first_column], packed, dist='NORM_B16')
            else:
                S.vsts(buffer[row, first_head, first_column], S.vcvt(first, T.bfloat16), dist='PK_B32')
                if has_second:
                    S.vsts(buffer[row, second_head, second_column], S.vcvt(second, T.bfloat16), dist='PK_B32')
        else:
            S.vsts(buffer[row, first_head, first_column], first)
            if has_second:
                S.vsts(buffer[row, second_head, second_column], second)

    @T.prim_func
    def _rope_kernel(
        x: T.StridedTensor((batch, seqlen, nheads, rotary_dim), (x_stride_b, x_stride_0, x_stride_1, 1), dtype),  # type: ignore
        cos_sin_cache: T.StridedTensor((seqlen_ro, rotary_dim), (cos_sin_stride, 1), T.float32),  # type: ignore
        positions: T.StridedTensor((batch, seqlen), (positions_stride_b, 1), positions_dtype),  # type: ignore
        seqlen_offsets: T.int32,
    ) -> None:
        T.assume(seqlen > 0)
        with T.Kernel(num_vec_cores) as core:
            x_ub = T.alloc_shared((block_tokens, nheads, rotary_dim), dtype)
            cos_sin_ub = T.alloc_shared((block_tokens, rotary_dim), T.float32)
            T.annotate_buffer_versions({x_ub: num_stages, cos_sin_ub: num_stages})

            for batch_id, token_block in T.Persistent(
                [batch, T.ceildiv(seqlen, block_tokens)], num_vec_cores, core, group_size=1, num_stages=num_stages
            ):
                token_start = token_block * block_tokens
                rows = T.min(block_tokens, seqlen - token_start)
                if can_bulk_copy:
                    T.copy(x[batch_id, token_start, 0, 0], x_ub[:rows, :, :], l2_cache_ctrl='NOTALLOC_KEEP')
                else:
                    for row in T.serial(block_tokens):
                        if row < rows:
                            T.copy(x[batch_id, token_start + row, 0, 0], x_ub[row, :, :], l2_cache_ctrl='NOTALLOC_KEEP')
                if has_position_tensor:
                    for row in T.serial(block_tokens):
                        if row < rows:
                            position = positions[batch_id, token_start + row] + seqlen_offsets
                            T.copy(cos_sin_cache[position, 0], cos_sin_ub[row, :])
                else:
                    position = token_start + seqlen_offsets
                    T.copy(cos_sin_cache[position, 0], cos_sin_ub[:rows, :])

                with T.SimdVF():
                    if rotary_dim == SIMD_LANES and interleaved:
                        cos_indices = S.vand(S.vci(0, T.int32), S.vdup(31, T.int32))
                        sin_indices = S.vor(cos_indices, S.vdup(32, T.int32))
                    for row in T.serial(block_tokens):
                        if rotary_dim == SIMD_LANES:
                            cos_sin_values = S.vld(cos_sin_ub[row, 0])
                            if interleaved:
                                cosine = S.vselr(cos_sin_values, cos_indices)
                                sine = S.vselr(cos_sin_values, sin_indices)
                            else:
                                cosine, sine = S.vintlv(cos_sin_values, cos_sin_values)
                        else:
                            cosine = S.vld(cos_sin_ub[row, 0])
                            sine = S.vld(cos_sin_ub[row, half_dim])
                        sine, neg_sine = (sine, S.vneg(sine)) if conjugate else (S.vneg(sine), sine)

                        if rotary_dim == SIMD_LANES:
                            for pair in range(nheads // 2):
                                head = pair * 2
                                packed_bf16 = interleaved and dtype == T.bfloat16
                                first, second = load2(x_ub, row, head, 0, head + 1, 0, packed_bf16)
                                if not packed_bf16:
                                    first, second = S.vdintlv(first, second) if interleaved else S.vintlv(first, second)
                                first, second = rotate(first, second, cosine, sine, neg_sine)
                                if not packed_bf16:
                                    first, second = S.vintlv(first, second) if interleaved else S.vdintlv(first, second)
                                store2(x_ub, row, head, 0, head + 1, 0, first, second, True, packed_bf16)
                            if nheads % 2:
                                odd_head = nheads - 1
                                first, second = load2(x_ub, row, odd_head, 0, odd_head, 0, False)
                                first, second = S.vdintlv(first, second) if interleaved else S.vintlv(first, second)
                                first, second = rotate(first, second, cosine, sine, neg_sine)
                                first, second = S.vintlv(first, second) if interleaved else S.vdintlv(first, second)
                                store2(x_ub, row, odd_head, 0, odd_head, 0, first, second, False, False)
                        else:
                            for head in range(nheads):
                                first, second = load2(x_ub, row, head, 0, head, half_dim, True)
                                if dtype == T.bfloat16:
                                    first, second = (first, second) if interleaved else S.vintlv(first, second)
                                elif interleaved:
                                    first, second = S.vdintlv(first, second)
                                first, second = rotate(first, second, cosine, sine, neg_sine)
                                if dtype == T.bfloat16:
                                    first, second = (first, second) if interleaved else S.vdintlv(first, second)
                                elif interleaved:
                                    first, second = S.vintlv(first, second)
                                store2(x_ub, row, head, 0, head, half_dim, first, second, True, True)
                if can_bulk_copy:
                    T.copy(x_ub[:rows, :, :], x[batch_id, token_start, 0, 0])
                else:
                    for row in T.serial(block_tokens):
                        if row < rows:
                            T.copy(x_ub[row, :, :], x[batch_id, token_start + row, 0, 0])

    return _rope_kernel

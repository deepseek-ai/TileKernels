from typing import Any

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=None, target='cuda')
def get_rope_kernel_cuda(
    dtype: T.dtype,
    positions_dtype: T.dtype,
    nheads: int,
    rotary_dim: int,
    has_position_tensor: bool,
    batch_is_one: bool,
    interleaved: bool,
    conjugate: bool,
    cos_sin_stride: int,
    x_stride_0: int,
    x_stride_1: int,
    use_pdl: bool = False,
) -> Any:
    assert dtype in (T.float16, T.bfloat16, T.float32)
    assert positions_dtype in (T.int32, T.int64)
    assert rotary_dim in (32, 64, 128, 256)

    batch = T.dynamic('batch')
    seqlen = T.dynamic('seqlen')
    seqlen_ro = T.dynamic('seqlen_ro')
    x_stride_b = T.dynamic('x_stride_b')
    positions_stride_b = T.dynamic('positions_stride_b')

    x_shape = (batch, seqlen, nheads, rotary_dim)
    x_strides = (x_stride_b, x_stride_0, x_stride_1, 1)
    positions_shape = (batch, seqlen)
    positions_strides = (positions_stride_b, 1)

    half_dim = rotary_dim // 2
    total_pairs = nheads * half_dim
    vec = 4
    threads_per_head = half_dim // vec
    threads = max(nheads, half_dim) // threads_per_head * threads_per_head
    pairs_per_step = threads * vec
    heads_per_thread = max(1, (total_pairs + pairs_per_step - 1) // pairs_per_step)
    # Multiple tokens per CTA and multiple heads per thread are mutually exclusive.
    tokens_per_cta = max(1, pairs_per_step // total_pairs)  # for small num_heads
    can_early_trigger_pdl = total_pairs >= 1024

    @T.prim_func
    def _rope_kernel(
        x: T.StridedTensor(x_shape, x_strides, dtype),  # type: ignore
        cos_sin_cache: T.StridedTensor(
            (seqlen_ro, rotary_dim),
            (cos_sin_stride, 1),
            T.float32,
        ),  # type: ignore
        positions: T.StridedTensor(positions_shape, positions_strides, positions_dtype),  # type: ignore
        seqlen_offsets: T.int32,
    ) -> None:
        with T.Kernel(T.ceildiv(seqlen, tokens_per_cta), 1 if batch_is_one else batch, threads=threads) as (token_block, batch_id):
            T.assume(x_stride_b % (vec * 2) == 0)
            batch_id = 0 if batch_is_one else batch_id
            x_local = T.alloc_local((heads_per_thread, vec * 2), dtype)
            first_local = T.alloc_local((heads_per_thread, vec), dtype)
            second_local = T.alloc_local((heads_per_thread, vec), dtype)
            cos_sin_local = T.alloc_local((vec * 2,), T.float32)

            thread_group = T.get_thread_binding() // threads_per_head
            vec_offset = T.get_thread_binding() % threads_per_head
            linear_head = thread_group * heads_per_thread
            head_offset = linear_head % nheads
            token_offset = linear_head // nheads
            token = token_block * tokens_per_cta + token_offset

            if token_offset < tokens_per_cta and token < seqlen:
                position = T.alloc_var(T.int32)
                position = seqlen_offsets
                if has_position_tensor:
                    position += positions[batch_id, token]
                else:
                    position += token

                for i in T.vectorized(vec):
                    cos_sin_local[i] = cos_sin_cache[position, vec_offset * vec + i]
                    cos_sin_local[vec + i] = cos_sin_cache[position, half_dim + vec_offset * vec + i]

                if use_pdl:
                    T.pdl_sync()

                for i in T.unroll(heads_per_thread):
                    if head_offset + i < nheads:
                        if interleaved:
                            for j in T.vectorized(vec * 2):
                                x_local[i, j] = x[batch_id, token, head_offset + i, vec_offset * (vec * 2) + j]
                            for j in T.unroll(vec):
                                first_local[i, j] = x_local[i, j * 2]
                                second_local[i, j] = x_local[i, j * 2 + 1]
                        else:
                            for j in T.vectorized(vec):
                                first_local[i, j] = x[batch_id, token, head_offset + i, vec_offset * vec + j]
                                second_local[i, j] = x[batch_id, token, head_offset + i, half_dim + vec_offset * vec + j]

                for i in T.unroll(heads_per_thread):
                    if head_offset + i < nheads:
                        for j in T.unroll(vec):
                            first = T.cast(first_local[i, j], T.float32)
                            second = T.cast(second_local[i, j], T.float32)
                            cosine = cos_sin_local[j]
                            sine = cos_sin_local[vec + j]
                            if conjugate:
                                first_local[i, j] = T.cast(T.ieee_fmaf(first, cosine, second * sine), dtype)
                                second_local[i, j] = T.cast(T.ieee_fmaf(first, -sine, second * cosine), dtype)
                            else:
                                first_local[i, j] = T.cast(T.ieee_fmaf(first, cosine, -(second * sine)), dtype)
                                second_local[i, j] = T.cast(T.ieee_fmaf(first, sine, second * cosine), dtype)

                if use_pdl and can_early_trigger_pdl:
                    T.pdl_trigger()

                for i in T.unroll(heads_per_thread):
                    if head_offset + i < nheads:
                        if interleaved:
                            for j in T.unroll(vec):
                                x_local[i, j * 2] = first_local[i, j]
                                x_local[i, j * 2 + 1] = second_local[i, j]
                            for j in T.vectorized(vec * 2):
                                x[batch_id, token, head_offset + i, vec_offset * (vec * 2) + j] = x_local[i, j]
                        else:
                            for j in T.vectorized(vec):
                                x[batch_id, token, head_offset + i, vec_offset * vec + j] = first_local[i, j]
                                x[batch_id, token, head_offset + i, half_dim + vec_offset * vec + j] = second_local[i, j]

    return _rope_kernel

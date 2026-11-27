import math

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from tile_kernels.engram.utils import load_f32
from tile_kernels.utils import align, ceil_div


@tilelang.jit
def get_engram_grad_w_reduce_kernel_asc(
    hidden_size: int,
    num_partials: int,
    hc_mult: int,
    has_value_norm: bool,
    dtype: T.dtype,
    num_vec_cores: int,
):
    vec_size = 64
    heads_per_task = 2
    assert dtype in (T.bfloat16, T.float32)
    assert hidden_size % (vec_size * heads_per_task) == 0
    assert hc_mult == 4
    assert num_vec_cores % 2 == 0

    blk_d = align(ceil_div(hidden_size * 2, num_vec_cores), 128)
    blk_d = math.gcd(hidden_size, blk_d)
    blk_v = blk_d // 2
    num_tiles = hidden_size // blk_d
    num_chunks = blk_d // vec_size
    num_v_chunks = blk_v // vec_size

    @T.prim_func
    def engram_grad_w_reduce_kernel_asc(
        grad_w_partial: T.Tensor[(num_partials, hc_mult + has_value_norm, hidden_size), T.float32],
        weight_hidden: T.Tensor[(hc_mult, hidden_size), dtype],
        weight_embed: T.Tensor[(hc_mult, hidden_size), dtype],
        grad_weight_hidden: T.Tensor[(hc_mult, hidden_size), T.float32],
        grad_weight_embed: T.Tensor[(hc_mult, hidden_size), T.float32],
        grad_weight_value: T.Tensor[(hidden_size,), T.float32],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            partial_ub = T.alloc_shared((heads_per_task, num_partials, blk_d), T.float32)
            weight_hidden_ub = T.alloc_shared((heads_per_task, blk_d), dtype)
            weight_embed_ub = T.alloc_shared((heads_per_task, blk_d), dtype)
            grad_hidden_ub = T.alloc_shared((heads_per_task, blk_d), T.float32)
            grad_embed_ub = T.alloc_shared((heads_per_task, blk_d), T.float32)
            if has_value_norm:
                partial_v_ub = T.alloc_shared((num_partials, blk_v), T.float32)
                grad_value_ub = T.alloc_shared((blk_v,), T.float32)

            for tile, head_pair in T.Persistent(
                [num_tiles, hc_mult // heads_per_task],
                num_vec_cores,
                core_id,
                group_size=1,
                num_stages=0,
            ):
                offset = tile * blk_d

                for local_head in T.unroll(heads_per_task, explicit=True):
                    head = head_pair * heads_per_task + local_head
                    T.copy(grad_w_partial[:, head, offset : offset + blk_d], partial_ub[local_head, :, :])
                    T.copy(weight_hidden[head, offset : offset + blk_d], weight_hidden_ub[local_head, :])
                    T.copy(weight_embed[head, offset : offset + blk_d], weight_embed_ub[local_head, :])
                    T.copy(grad_weight_hidden[head, offset : offset + blk_d], grad_hidden_ub[local_head, :])
                    T.copy(grad_weight_embed[head, offset : offset + blk_d], grad_embed_ub[local_head, :])

                if has_value_norm:
                    v_offset = offset + head_pair * blk_v
                    T.copy(grad_w_partial[:, hc_mult, v_offset : v_offset + blk_v], partial_v_ub)
                    T.copy(grad_weight_value[v_offset : v_offset + blk_v], grad_value_ub)

                with T.SimdVF():
                    zero = S.vdup(0.0, T.float32)
                    accum = S.alloc_local((heads_per_task, num_chunks), T.float32)
                    for local_head in T.unroll(heads_per_task, explicit=True):
                        for chunk in T.unroll(num_chunks, explicit=True):
                            accum[local_head, chunk] = zero
                    if has_value_norm:
                        value_accum = S.alloc_local((num_v_chunks,), T.float32)
                        for chunk in T.unroll(num_v_chunks, explicit=True):
                            value_accum[chunk] = zero

                    for partial in T.serial(num_partials):
                        for local_head in T.unroll(heads_per_task, explicit=True):
                            for chunk in T.unroll(num_chunks, explicit=True):
                                partial_value = S.vld(partial_ub[local_head, partial, chunk * vec_size])
                                accum[local_head, chunk] = S.vadd(accum[local_head, chunk], partial_value)
                        if has_value_norm:
                            for chunk in T.unroll(num_v_chunks, explicit=True):
                                value_accum[chunk] = S.vadd(value_accum[chunk], S.vld(partial_v_ub[partial, chunk * vec_size]))

                    grad_acc = S.alloc_local((2,), T.float32)
                    for local_head in T.unroll(heads_per_task, explicit=True):
                        for chunk in T.unroll(num_chunks, explicit=True):
                            chunk_offset = chunk * vec_size
                            weight_hidden_f32 = load_f32(weight_hidden_ub[local_head, chunk_offset])
                            weight_embed_f32 = load_f32(weight_embed_ub[local_head, chunk_offset])
                            grad_acc[0] = S.vld(grad_hidden_ub[local_head, chunk_offset])
                            grad_acc[1] = S.vld(grad_embed_ub[local_head, chunk_offset])
                            S.vmula(grad_acc[0], accum[local_head, chunk], weight_embed_f32)
                            S.vmula(grad_acc[1], accum[local_head, chunk], weight_hidden_f32)
                            S.vsts(grad_hidden_ub[local_head, chunk_offset], grad_acc[0])
                            S.vsts(grad_embed_ub[local_head, chunk_offset], grad_acc[1])
                    if has_value_norm:
                        for chunk in T.unroll(num_v_chunks, explicit=True):
                            chunk_offset = chunk * vec_size
                            grad_value = S.vld(grad_value_ub[chunk_offset])
                            S.vsts(grad_value_ub[chunk_offset], S.vadd(grad_value, value_accum[chunk]))

                for local_head in T.unroll(heads_per_task, explicit=True):
                    head = head_pair * heads_per_task + local_head
                    T.copy(grad_hidden_ub[local_head, :], grad_weight_hidden[head, offset : offset + blk_d])
                    T.copy(grad_embed_ub[local_head, :], grad_weight_embed[head, offset : offset + blk_d])
                if has_value_norm:
                    T.copy(grad_value_ub, grad_weight_value[v_offset : v_offset + blk_v])

    return engram_grad_w_reduce_kernel_asc

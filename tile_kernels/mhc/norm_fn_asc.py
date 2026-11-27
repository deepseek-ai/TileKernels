import tilelang
from tilelang.ascend import language as T

from tile_kernels.config import get_max_ub_per_vector_core

_COMPACT_REDUCE_TOKEN_BLOCK = 16
_COMPACT_REDUCE_SPLIT_CHUNK = 32
_COMPACT_REDUCE_MHC_MULT3 = 24
_COMPACT_REDUCE_VREGS = 6


@tilelang.jit
def mhc_fn_normw_merge_fwd_asc(
    fn,
    normw,
    out_fn,
    mhc_mult3: int,
    hidden_block: int,
    num_stages: int,
    num_vec_cores: int,
):
    from tilelang.ascend.language import simd as S

    hidden = T.const('hidden')
    fn: T.Tensor[(mhc_mult3, hidden), T.float32]
    normw: T.Tensor[(hidden,), T.float32]
    out_fn: T.Tensor[(mhc_mult3, hidden), T.float32]

    assert hidden % hidden_block == 0, 'Ascend fn/normw merge requires hidden divisible by hidden_block'
    assert hidden_block % 128 == 0, 'Ascend fn/normw merge SIMD path requires hidden_block divisible by 128'

    num_hidden_blocks = hidden // hidden_block

    with T.Kernel(num_vec_cores) as core_id:
        fn_ub = T.alloc_shared((mhc_mult3, hidden_block), T.float32)
        normw_ub = T.alloc_shared((hidden_block,), T.float32)
        out_fn_ub = T.alloc_shared((mhc_mult3, hidden_block), T.float32)
        T.annotate_buffer_versions(
            {
                fn_ub: num_stages,
                normw_ub: num_stages,
                out_fn_ub: num_stages,
            }
        )

        for block_id in T.Persistent([num_hidden_blocks], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            hidden_offset = block_id * hidden_block
            T.copy(fn[:mhc_mult3, hidden_offset : hidden_offset + hidden_block], fn_ub)
            T.copy(normw[hidden_offset : hidden_offset + hidden_block], normw_ub)

            with T.SimdVF():
                mask = S.pset(32)
                for vec_pair in range(hidden_block // 128):
                    col0 = vec_pair * 128
                    col1 = col0 + 64
                    normw0 = S.vld(normw_ub[col0])
                    normw1 = S.vld(normw_ub[col1])
                    for row in range(mhc_mult3):
                        fn0 = S.vld(fn_ub[row, col0])
                        fn1 = S.vld(fn_ub[row, col1])
                        S.vsts(out_fn_ub[row, col0], S.vmul(fn0, normw0, mask), mask, 'NORM_B32')
                        S.vsts(out_fn_ub[row, col1], S.vmul(fn1, normw1, mask), mask, 'NORM_B32')

            T.copy(out_fn_ub, out_fn[:mhc_mult3, hidden_offset : hidden_offset + hidden_block])


@tilelang.jit
def mhc_fn_normw_merge_bwd_asc(
    fn,
    normw,
    out_fn_grad,
    fn_grad,
    normw_grad,
    mhc_mult3: int,
    hidden_block: int,
    num_stages: int,
    num_vec_cores: int,
):
    from tilelang.ascend.language import simd as S

    hidden = T.const('hidden')
    fn: T.Tensor[(mhc_mult3, hidden), T.float32]
    normw: T.Tensor[(hidden,), T.float32]
    out_fn_grad: T.Tensor[(mhc_mult3, hidden), T.float32]
    fn_grad: T.Tensor[(mhc_mult3, hidden), T.float32]
    normw_grad: T.Tensor[(hidden,), T.float32]

    assert hidden % hidden_block == 0, 'Ascend fn/normw merge requires hidden divisible by hidden_block'
    assert hidden_block % 128 == 0, 'Ascend fn/normw merge SIMD path requires hidden_block divisible by 128'

    num_hidden_blocks = hidden // hidden_block

    with T.Kernel(num_vec_cores) as core_id:
        fn_ub = T.alloc_shared((mhc_mult3, hidden_block), T.float32)
        normw_ub = T.alloc_shared((hidden_block,), T.float32)
        out_fn_grad_ub = T.alloc_shared((mhc_mult3, hidden_block), T.float32)
        fn_grad_ub = T.alloc_shared((mhc_mult3, hidden_block), T.float32)
        normw_grad_ub = T.alloc_shared((hidden_block,), T.float32)
        T.annotate_buffer_versions(
            {
                fn_ub: num_stages,
                normw_ub: num_stages,
                out_fn_grad_ub: num_stages,
                fn_grad_ub: num_stages,
                normw_grad_ub: num_stages,
            }
        )

        for block_id in T.Persistent([num_hidden_blocks], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            hidden_offset = block_id * hidden_block
            T.copy(fn[:mhc_mult3, hidden_offset : hidden_offset + hidden_block], fn_ub)
            T.copy(normw[hidden_offset : hidden_offset + hidden_block], normw_ub)
            T.copy(out_fn_grad[:mhc_mult3, hidden_offset : hidden_offset + hidden_block], out_fn_grad_ub)
            T.copy(fn_grad[:mhc_mult3, hidden_offset : hidden_offset + hidden_block], fn_grad_ub)
            T.copy(normw_grad[hidden_offset : hidden_offset + hidden_block], normw_grad_ub)

            with T.SimdVF():
                mask = S.pset(32)
                for vec_pair in range(hidden_block // 128):
                    col0 = vec_pair * 128
                    col1 = col0 + 64
                    normw0 = S.vld(normw_ub[col0])
                    normw1 = S.vld(normw_ub[col1])
                    normw_grad0 = S.alloc_var(T.float32)
                    normw_grad1 = S.alloc_var(T.float32)
                    normw_grad0 = S.vld(normw_grad_ub[col0])
                    normw_grad1 = S.vld(normw_grad_ub[col1])

                    for row in range(mhc_mult3):
                        out_grad0 = S.vld(out_fn_grad_ub[row, col0])
                        out_grad1 = S.vld(out_fn_grad_ub[row, col1])
                        fn0 = S.vld(fn_ub[row, col0])
                        fn1 = S.vld(fn_ub[row, col1])
                        fn_grad0 = S.vld(fn_grad_ub[row, col0])
                        fn_grad1 = S.vld(fn_grad_ub[row, col1])

                        fn_grad_out0 = S.vadd(fn_grad0, S.vmul(out_grad0, normw0, mask), mask)
                        fn_grad_out1 = S.vadd(fn_grad1, S.vmul(out_grad1, normw1, mask), mask)
                        normw_grad0 = S.vadd(normw_grad0, S.vmul(out_grad0, fn0, mask), mask)
                        normw_grad1 = S.vadd(normw_grad1, S.vmul(out_grad1, fn1, mask), mask)

                        S.vsts(fn_grad_ub[row, col0], fn_grad_out0, mask, 'NORM_B32')
                        S.vsts(fn_grad_ub[row, col1], fn_grad_out1, mask, 'NORM_B32')

                    S.vsts(normw_grad_ub[col0], normw_grad0, mask, 'NORM_B32')
                    S.vsts(normw_grad_ub[col1], normw_grad1, mask, 'NORM_B32')

            T.copy(fn_grad_ub, fn_grad[:mhc_mult3, hidden_offset : hidden_offset + hidden_block])
            T.copy(normw_grad_ub, normw_grad[hidden_offset : hidden_offset + hidden_block])


def _compact_reduce_split_chunk(n_splits: int, token_block: int, num_stages: int) -> int:
    fixed_ub = (3 * token_block * _COMPACT_REDUCE_MHC_MULT3 + 2 * 64) * 4
    bytes_per_split = (token_block * _COMPACT_REDUCE_MHC_MULT3 + 64) * 4 * num_stages
    max_chunk = (get_max_ub_per_vector_core(use_simt=False) - fixed_ub) // bytes_per_split
    for split_chunk in (_COMPACT_REDUCE_SPLIT_CHUNK, 16, 8, 4, 2):
        if split_chunk <= max_chunk and n_splits % split_chunk == 0:
            return split_chunk
    return 1


@tilelang.jit
def mhc_reduce_partials_and_rmsnorm_fwd_asc(
    out_mul_splitted,
    sqrsum_splitted,
    out_mul,
    sqrsum,
    out,
    rms_group_size: int,
    rms_eps: float,
    n_splits: int,
    num_vec_cores: int,
    token_block: int = _COMPACT_REDUCE_TOKEN_BLOCK,
):
    from tilelang.ascend.language import simd as S

    num_tokens = T.dynamic('num_tokens')
    mhc_mult3 = T.const('mhc_mult3')
    n_rms_group = T.const('n_rms_group')
    assert n_rms_group == 1, 'Ascend reduce/rmsnorm fwd currently supports one RMS group'
    assert mhc_mult3 == _COMPACT_REDUCE_MHC_MULT3, 'Ascend compact reduce assumes mhc_mult3=24'

    out_mul_splitted: T.Tensor[(n_splits, num_tokens, n_rms_group, mhc_mult3), T.float32]
    sqrsum_splitted: T.Tensor[(n_splits, num_tokens, n_rms_group), T.float32]
    out_mul: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32]
    sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32]
    out: T.Tensor[(num_tokens, mhc_mult3), T.float32]

    assert 1 <= token_block <= 64 and token_block * mhc_mult3 % 64 == 0, (
        f'token_block must be in 1..64 with token_block * mhc_mult3 divisible by 64, got {token_block}'
    )
    compact_vregs = token_block * mhc_mult3 // 64
    token_mask = 'PAT_ALL' if token_block == 64 else f'PAT_VL{token_block}'
    sq_pad64 = 64
    num_stages = 3
    split_chunk = _compact_reduce_split_chunk(n_splits, token_block, num_stages)
    n_token_blocks = T.ceildiv(num_tokens, token_block)
    n_split_chunks = n_splits // split_chunk
    assert n_splits % split_chunk == 0, 'compact reduce requires n_splits divisible by selected split_chunk'

    with T.Kernel(num_vec_cores) as core_id:
        mul_ub = T.alloc_shared((split_chunk, token_block, mhc_mult3), T.float32)
        sq_ub = T.alloc_shared((split_chunk, sq_pad64), T.float32)
        acc_ub = T.alloc_shared((compact_vregs, 64), T.float32)
        sq_acc_ub = T.alloc_shared(64, T.float32)
        mul_out_ub = T.alloc_shared((token_block, mhc_mult3), T.float32)
        out_ub = T.alloc_shared((token_block, mhc_mult3), T.float32)
        sq_out_ub = T.alloc_shared(64, T.float32)
        T.annotate_buffer_versions(
            {
                mul_ub: num_stages,
                sq_ub: num_stages,
                acc_ub: 1,
                sq_acc_ub: 1,
                mul_out_ub: 1,
                out_ub: 1,
                sq_out_ub: 1,
            }
        )

        for block_id in T.Persistent([n_token_blocks], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            token_base = block_id * token_block
            # Ragged tail blocks are handled by the same vectorized path:
            # T.copy clamps out-of-bound GM reads/writes, and the per-token
            # lane masks keep the padded tokens' stale UB lanes isolated,
            # so no separate scalar tail path is needed.
            with T.SimdVF():
                mask_full = S.pset(32)
                zero = S.vdup(T.float32(0.0), T.float32, mask_full)
                for vec_idx in T.unroll(compact_vregs, explicit=True):
                    S.vsts(acc_ub[vec_idx, 0], zero, mask_full, 'NORM_B32')
                S.vsts(sq_acc_ub[0], zero, mask_full, 'NORM_B32')
                S.mem_bar('VST_VLD')

            for split_chunk_idx in T.serial(n_split_chunks):
                split_base = split_chunk_idx * split_chunk
                T.copy(
                    out_mul_splitted[split_base : split_base + split_chunk, token_base : token_base + token_block, 0, :mhc_mult3],
                    mul_ub,
                )
                T.copy(
                    sqrsum_splitted[split_base : split_base + split_chunk, token_base : token_base + token_block, 0],
                    sq_ub[:, :token_block],
                )

                with T.SimdVF():
                    mask_full = S.pset(32)
                    mask_token = S.pset(32, token_mask)
                    acc = S.alloc_local((compact_vregs,), T.float32)
                    acc_sq = S.alloc_var(T.float32)
                    for vec_idx in T.unroll(compact_vregs, explicit=True):
                        acc[vec_idx] = S.vld(acc_ub[vec_idx, 0])
                    acc_sq = S.vld(sq_acc_ub[0])

                    for local_split in range(split_chunk):
                        for vec_idx in T.unroll(compact_vregs, explicit=True):
                            acc[vec_idx] = S.vadd(
                                acc[vec_idx], S.vld(mul_ub[local_split, (vec_idx * 64) // mhc_mult3, (vec_idx * 64) % mhc_mult3]), mask_full
                            )
                        acc_sq = S.vadd(acc_sq, S.vld(sq_ub[local_split, 0]), mask_token)

                    for vec_idx in T.unroll(compact_vregs, explicit=True):
                        S.vsts(acc_ub[vec_idx, 0], acc[vec_idx], mask_full, 'NORM_B32')
                    S.vsts(sq_acc_ub[0], acc_sq, mask_full, 'NORM_B32')
                    S.mem_bar('VST_VLD')

            with T.SimdVF():
                mask_full = S.pset(32)
                inv_rms_group_size = S.vdup(T.float32(1.0 / rms_group_size), T.float32, mask_full)
                one = S.vdup(T.float32(1.0), T.float32, mask_full)
                zero = S.vdup(T.float32(0.0), T.float32, mask_full)
                lane_idx = S.vci(T.int32(0), T.int32)
                final_sq_acc = S.alloc_var(T.float32)
                final_sq_acc = S.vld(sq_acc_ub[0])
                S.vsts(sq_out_ub[0], final_sq_acc, mask_full, 'NORM_B32')
                S.mem_bar('VST_VLD')

                for vec_idx in T.unroll(compact_vregs, explicit=True):
                    acc_vec = S.vld(acc_ub[vec_idx, 0])
                    rms_vec = S.alloc_var(T.float32)
                    rms_vec = zero
                    elem_start = vec_idx * 64
                    elem_end = elem_start + 64
                    token_start = elem_start // _COMPACT_REDUCE_MHC_MULT3
                    token_end = (elem_end - 1) // _COMPACT_REDUCE_MHC_MULT3
                    for token_idx in range(token_start, token_end + 1):
                        sq_val = S.vld(sq_out_ub[token_idx], dist='BRC_B32')
                        rms_val = S.vdiv(
                            one, S.vsqrt(S.vadds(S.vmul(sq_val, inv_rms_group_size, mask_full), T.float32(rms_eps), mask_full), mask_full), mask_full
                        )
                        lane_start = token_idx * _COMPACT_REDUCE_MHC_MULT3 - elem_start
                        lane_end = lane_start + _COMPACT_REDUCE_MHC_MULT3
                        token_mask = S.pand(
                            S.vcmps(lane_idx, T.int32(lane_start), mask_full, 'ge'),
                            S.vcmps(lane_idx, T.int32(lane_end), mask_full, 'lt'),
                            mask_full,
                        )
                        rms_vec = S.vsel(rms_val, rms_vec, token_mask)
                    S.vsts(mul_out_ub[(vec_idx * 64) // mhc_mult3, (vec_idx * 64) % mhc_mult3], acc_vec, mask_full, 'NORM_B32')
                    S.vsts(
                        out_ub[(vec_idx * 64) // mhc_mult3, (vec_idx * 64) % mhc_mult3], S.vmul(acc_vec, rms_vec, mask_full), mask_full, 'NORM_B32'
                    )

            T.copy(mul_out_ub, out_mul[token_base : token_base + token_block, 0, :])
            T.copy(out_ub, out[token_base : token_base + token_block, :])
            T.copy(sq_out_ub[:token_block], sqrsum[token_base : token_base + token_block, 0])


@tilelang.jit
def mhc_reduce_partials_and_rmsnorm_bwd_asc(
    out_grad,
    out_mul,
    sqrsum,
    out_mul_grad,
    sqrsum_grad,
    rms_group_size: int,
    rms_eps: float,
    num_vec_cores: int,
):
    from tilelang.ascend.language import simd as S

    num_tokens = T.dynamic('num_tokens')
    mhc_mult3 = T.const('mhc_mult3')
    n_rms_group = T.const('n_rms_group')
    assert n_rms_group == 1, 'Ascend reduce/rmsnorm bwd currently supports one RMS group'
    assert mhc_mult3 <= 32, f'Ascend reduce/rmsnorm bwd SimtVF maps mhc_mult3 to one 32-lane vector, got {mhc_mult3}'

    out_grad: T.Tensor[(num_tokens, mhc_mult3), T.float32]
    out_mul: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32]
    sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32]
    out_mul_grad: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32]
    sqrsum_grad: T.Tensor[(num_tokens, n_rms_group), T.float32]

    token_block = 16
    mhc_mult3_pad32 = 32
    num_stages = 6
    n_token_blocks = T.ceildiv(num_tokens, token_block)

    with T.Kernel(num_vec_cores) as core_id:
        out_grad_ub = T.alloc_shared((token_block, mhc_mult3_pad32), T.float32)
        out_mul_ub = T.alloc_shared((token_block, mhc_mult3_pad32), T.float32)
        out_mul_grad_ub = T.alloc_shared((token_block, mhc_mult3_pad32), T.float32)
        sqrsum_ub = T.alloc_shared(token_block, T.float32)
        sqrsum_grad_ub = T.alloc_shared((token_block, 8), T.float32)
        sqrsum_grad_compact_ub = T.alloc_shared(64, T.float32)
        T.annotate_buffer_versions(
            {
                out_grad_ub: num_stages,
                out_mul_ub: num_stages,
                out_mul_grad_ub: num_stages,
                sqrsum_ub: num_stages,
                sqrsum_grad_ub: num_stages,
                sqrsum_grad_compact_ub: num_stages,
            }
        )

        for block_id in T.Persistent([n_token_blocks], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            token_base = block_id * token_block
            T.copy(out_grad[token_base : token_base + token_block, :mhc_mult3], out_grad_ub[:, :mhc_mult3])
            T.copy(out_mul[token_base : token_base + token_block, 0, :mhc_mult3], out_mul_ub[:, :mhc_mult3])
            T.copy(sqrsum[token_base : token_base + token_block, 0], sqrsum_ub[:token_block])

            with T.SimdVF():
                mask_full = S.pset(32)
                mask_lane0 = S.pset(32, 'PAT_VL1')
                mask_first_token = S.pset(32, 'PAT_VL32')
                lane_idx = S.vci(T.int32(0), T.int32)
                first_token_mask = S.vcmps(lane_idx, T.int32(mhc_mult3), mask_full, 'lt')
                second_token_mask = S.pand(
                    S.vcmps(lane_idx, T.int32(32), mask_full, 'ge'),
                    S.vcmps(lane_idx, T.int32(32 + mhc_mult3), mask_full, 'lt'),
                    mask_full,
                )
                inv_rms_group_size = S.vdup(T.float32(1.0 / rms_group_size), T.float32, mask_full)
                one = S.vdup(T.float32(1.0), T.float32, mask_full)
                neg_half = S.vdup(T.float32(-0.5), T.float32, mask_full)

                for pair_idx in T.unroll(token_block // 2, explicit=True):
                    grad_pair = S.vld(out_grad_ub[pair_idx * 2, 0])
                    mul_pair = S.vld(out_mul_ub[pair_idx * 2, 0])
                    sq_even = S.vld(sqrsum_ub[pair_idx * 2], dist='BRC_B32')
                    sq_odd = S.vld(sqrsum_ub[pair_idx * 2 + 1], dist='BRC_B32')
                    rms_even = S.vdiv(
                        one, S.vsqrt(S.vadds(S.vmul(sq_even, inv_rms_group_size, mask_full), T.float32(rms_eps), mask_full), mask_full), mask_full
                    )
                    rms_odd = S.vdiv(
                        one, S.vsqrt(S.vadds(S.vmul(sq_odd, inv_rms_group_size, mask_full), T.float32(rms_eps), mask_full), mask_full), mask_full
                    )
                    rms_pair = S.vsel(rms_even, rms_odd, mask_first_token)
                    grad_mul = S.vmul(grad_pair, mul_pair, mask_full)
                    out_mul_grad_pair = S.vmul(grad_pair, rms_pair, mask_full)
                    S.vsts(out_mul_grad_ub[pair_idx * 2, 0], out_mul_grad_pair, mask_full, 'NORM_B32')

                    rms_grad_even = S.vcadd(grad_mul, first_token_mask)
                    rms_grad_odd = S.vcadd(grad_mul, second_token_mask)
                    denom_even = S.vadds(sq_even, T.float32(rms_eps * rms_group_size), mask_full)
                    denom_odd = S.vadds(sq_odd, T.float32(rms_eps * rms_group_size), mask_full)
                    sq_grad_even = S.vmul(S.vdiv(S.vmul(rms_grad_even, rms_even, mask_full), denom_even, mask_full), neg_half, mask_full)
                    sq_grad_odd = S.vmul(S.vdiv(S.vmul(rms_grad_odd, rms_odd, mask_full), denom_odd, mask_full), neg_half, mask_full)
                    S.vsts(sqrsum_grad_ub[pair_idx * 2, 0], sq_grad_even, mask_lane0, 'ONEPT_B32')
                    S.vsts(sqrsum_grad_ub[pair_idx * 2 + 1, 0], sq_grad_odd, mask_lane0, 'ONEPT_B32')

                S.mem_bar('VST_VLD')
                sq_grad_compact = S.alloc_var(T.float32)
                sq_grad_compact = S.vdup(T.float32(0.0), T.float32, mask_full)
                for token_offset in T.unroll(token_block, explicit=True):
                    token_sq_grad = S.vld(sqrsum_grad_ub[token_offset, 0], dist='BRC_B32')
                    token_mask = S.vcmps(lane_idx, T.int32(token_offset), mask_full, 'eq')
                    sq_grad_compact = S.vsel(token_sq_grad, sq_grad_compact, token_mask)
                S.vsts(sqrsum_grad_compact_ub[0], sq_grad_compact, mask_full, 'NORM_B32')

            T.copy(out_mul_grad_ub[:, :mhc_mult3], out_mul_grad[token_base : token_base + token_block, 0, :mhc_mult3])
            T.copy(sqrsum_grad_compact_ub[:token_block], sqrsum_grad[token_base : token_base + token_block, 0])

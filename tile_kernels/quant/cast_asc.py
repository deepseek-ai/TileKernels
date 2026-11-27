import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S
from tile_kernels.config import get_max_ub_per_vector_core
from tile_kernels.quant.stochastic_rounding import rbits_hash_bf16_asc


@tilelang.jit()
def get_cast_kernel_asc(
    in_dtype: T.dtype,
    out_dtype: T.dtype,
    stochastic_rounding: bool,
    n_hint: int,
    num_vec_cores: int,
):
    n = T.dynamic('n', T.int64)
    max_bytes = max(in_dtype.bytes, out_dtype.bytes)
    # 256B for one vec; 4 vecs needed because 4 numbers to hash.
    vec_len = 1024 // max_bytes
    block_alignment = 8
    reps = get_max_ub_per_vector_core(use_simt=False) // (2 * 1024) // block_alignment * block_alignment

    block_size = vec_len * reps
    half_size = block_size // 2
    half_reps = reps // 2

    # stochastic round x1(0-255B), x2(256-512B) to bf16, using rbits. finite saturation not needed.
    @T.macro
    def stochastic_round(x1, x2, rbits):
        hi = S.alloc_var(T.uint16)
        lo, hi = S.vdintlv(T.reinterpret(x1, 'uint16x128'), T.reinterpret(x2, 'uint16x128'))
        should_carry = S.vcmp(T.reinterpret(rbits, 'uint16x128'), lo, op='lt')
        hi = S.vadds(hi, T.uint16(1), should_carry, mode='MODE_MERGING')
        return T.reinterpret(hi, 'bfloat16x128')

    # process 1024B elements a time -> vectors arith in hash is 256B wide
    @T.macro
    def process_chunk(in_chunk, base_e):
        h = rbits_hash_bf16_asc(in_chunk, base_e)
        res1 = stochastic_round(in_chunk[0], in_chunk[1], h[0])
        res2 = stochastic_round(in_chunk[2], in_chunk[3], h[1])
        return res1, res2

    # Convert half_size of fp32 values into bf16 and write into y_ub.
    @T.macro
    def convert_half(x_ub, y_ub, dest, base_id):
        with T.SimdVF():
            in_chunk = S.alloc_local((4,), T.float32)
            # Disable outer-loop unrolling to avoid code bloat and slowdown on CANN 9.3.
            for i in T.unroll(half_reps, unroll_factor=1):
                offset = i * vec_len
                for j in T.unroll(4):
                    in_chunk[j] = S.vld(x_ub[offset + j * vec_len // 4])
                res1, res2 = process_chunk(in_chunk, T.uint32((base_id + offset) // 4))
                S.vsts(y_ub[dest + offset], res1)
                S.vsts(y_ub[dest + offset + vec_len // 2], res2)

    # process a block: copyin x -> front_ub + back_ub; casted bf16 -> front_ub; (read next loop using back_ub); copyout front_ub -> y
    @T.macro
    def process_block(x, y, front_ub, back_ub, base_id, valid_len):
        # bf16 is only half the size of fp32, so it fits in the frist half input (front_ub).
        output = T.view(front_ub, (block_size,), out_dtype)
        front_len = T.min(half_size, valid_len)
        back_len = valid_len - half_size

        T.copy(x[base_id], front_ub[:front_len])

        if back_len > 0:
            T.copy(x[base_id + half_size], back_ub[:back_len])
        convert_half(front_ub, output, 0, base_id)
        if back_len > 0:
            convert_half(back_ub, output, half_size, base_id + half_size)

        T.copy(output[:valid_len], y[base_id])

    @T.prim_func
    def cast_kernel_asc(
        x: T.Tensor[(n,), in_dtype],
        y: T.Tensor[(n,), out_dtype],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            T.assume(n % n_hint == 0)
            # Two independent pairs of half-buffers, 240 KB in total.
            front0_ub = T.alloc_shared((half_size,), in_dtype)
            back0_ub = T.alloc_shared((half_size,), in_dtype)
            front1_ub = T.alloc_shared((half_size,), in_dtype)
            back1_ub = T.alloc_shared((half_size,), in_dtype)

            full_groups = n // (block_size * 4)
            for pid in T.Persistent([full_groups], num_vec_cores, core_id, group_size=1, num_stages=1):
                base_id = T.int64(pid) * block_size * 4
                # Stage(0) here is to preserve order of operations in a pipeline
                with T.Stage(0):
                    process_block(x, y, front0_ub, back0_ub, base_id, block_size)
                    process_block(x, y, front1_ub, back1_ub, base_id + block_size, block_size)
                    # Swap roles: the released back buffers receive the front half of the next block.
                    process_block(x, y, back0_ub, front0_ub, base_id + block_size * 2, block_size)
                    process_block(x, y, back1_ub, front1_ub, base_id + block_size * 3, block_size)

            # move tail blocks handling outside the main loop to reduce ifs in the main loop.
            tail_blocks = T.ceildiv(n, block_size) - full_groups * 4
            for tail_id in T.serial(core_id, tail_blocks, num_vec_cores):
                base_id = (full_groups * 4 + T.int64(tail_id)) * block_size
                process_block(x, y, front0_ub, back0_ub, base_id, T.min(block_size, n - base_id))

    return cast_kernel_asc

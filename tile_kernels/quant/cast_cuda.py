import tilelang
from tilelang import language as T

from tile_kernels.quant.stochastic_rounding import rbits_hash_cuda


@tilelang.jit(
    target='cuda',
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True,
    },
)
def get_cast_kernel_cuda(in_dtype: T.dtype, out_dtype: T.dtype, stochastic_rounding: bool = False, n_hint: int = 1):
    n = T.dynamic('n', T.int64)
    num_threads = 256
    num_rbits = out_dtype.bytes
    # we want read/writes to be 32B/thread. so vec_len = 32 // max_bytes
    max_bytes = max(in_dtype.bytes, out_dtype.bytes)
    vec_len = 32 // max_bytes
    block_size = num_threads * vec_len
    num_blocks = T.ceildiv(n, block_size)

    @T.prim_func
    def cast_kernel(
        x: T.Tensor[(n,), in_dtype],
        y: T.Tensor[(n,), out_dtype],
    ):
        with T.Kernel(num_blocks, threads=num_threads) as bid:
            T.assume(n % n_hint == 0)
            base_id = block_size * T.int64(bid)
            in_frag = T.alloc_fragment((block_size,), in_dtype)
            out_frag = T.alloc_fragment((block_size,), out_dtype)

            T.annotate_layout(
                {
                    in_frag: T.Fragment((block_size,), lambda i: (i // vec_len, i % vec_len)),
                    out_frag: T.Fragment((block_size,), lambda i: (i // vec_len, i % vec_len)),
                }
            )

            T.copy(x[base_id], in_frag)

            if not stochastic_rounding:
                for i in T.Parallel(block_size):
                    out_frag[i] = T.cast(in_frag[i], out_dtype)
            else:
                # hash generation needs 4 values a time, and cvt.rs requires f32.
                in_local = T.alloc_local((4,), T.float32)
                for k in T.Parallel(block_size // 4):
                    for i in T.vectorized(4):
                        in_local[i] = T.cast(in_frag[k * 4 + i], T.float32)
                    rbits = rbits_hash_cuda(in_local, T.uint32(base_id // 4 + k), num_rbits)
                    # all rbits together handles 4 elements. so one rbit -> (4 / num_rbits) elements
                    for i in T.unroll(num_rbits):
                        for j in T.vectorized(4 // num_rbits):
                            offset = i * (4 // num_rbits) + j
                            pos = k * 4 + offset
                            out_frag[pos] = T.cast(in_local[offset], out_dtype, round='rs', rbits=rbits[i])

            T.copy(out_frag, y[base_id])

    return cast_kernel

import tilelang
from tilelang import language as T
from tile_kernels.config import get_tilelang_target


@tilelang.jit(
    target=get_tilelang_target(),
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

    # gen random hash for rounding, according to doc
    # out = bf16 -> needs 2 hash numbers; out = fp8 -> only 1 hash is needed
    @T.macro
    def rbits_hash(a: T.uint32, b: T.uint32, c: T.uint32, d: T.uint32, e: T.uint32):
        h = T.alloc_var(T.uint32)
        h_res = T.alloc_local((num_rbits,), T.uint32)
        h = a * T.uint32(0x5671D42B) + b * T.uint32(0x9995E499) + c * T.uint32(0xACE1B8A5) + d * T.uint32(0xE153538D) + e
        h = h ^ (h >> 23)
        h = h * T.uint32(0x7FEB352D)
        h = h ^ (h >> 16)
        h_res[0] = h * T.uint32(0x846CA68B)
        h_res[0] = h_res[0] ^ (h_res[0] >> 11)
        # for bf16, we need 2 hash numbers
        if num_rbits == 2:
            h_res[1] = h * T.uint32(0xD35A2D97)
            h_res[1] = h_res[1] ^ (h_res[1] >> 11)
        return h_res

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

                    rbits = rbits_hash(
                        T.reinterpret(in_local[0], T.uint32),
                        T.reinterpret(in_local[1], T.uint32),
                        T.reinterpret(in_local[2], T.uint32),
                        T.reinterpret(in_local[3], T.uint32),
                        T.uint32(base_id // 4 + k),
                    )

                    # all rbits together handles 4 elements. so one rbit -> (4 / num_rbits) elements
                    for i in T.unroll(num_rbits):
                        for j in T.vectorized(4 // num_rbits):
                            offset = i * (4 // num_rbits) + j
                            pos = k * 4 + offset
                            out_frag[pos] = T.cast(in_local[offset], out_dtype, round='rs', rbits=rbits[i])

            T.copy(out_frag, y[base_id])

    return cast_kernel

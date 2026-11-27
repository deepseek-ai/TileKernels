import math

import tilelang
from tilelang import language as T

from tile_kernels.quant.common import *
from tile_kernels.utils import align


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True,
    },
)
def get_cast_back_kernel_cuda(
    hidden: int,
    in_config: CastInputConfig,
    out_dtype: T.dtype = T.bfloat16,
    use_pdl: bool = False,
):
    num_threads = 128
    num_elems_per_block = 8192
    num_per_tokens, num_per_channels = in_config.sf_block

    if num_per_tokens == 1:
        # Default case: 1 * gcd(hidden, 8192)
        TILE_K = math.gcd(hidden, num_elems_per_block)

        # Replication optimization case: 1 * hidden
        if hidden <= num_elems_per_block:
            TILE_K = align(hidden, num_threads * (2 if in_config.dtype == T.float4_e2m1fn else 1))

        TILE_M = num_elems_per_block // TILE_K

        # When TILE_M is small, set TILE_M = 1 to avoid predicated loads for better performance
        if TILE_M <= 3:
            TILE_M = 1
    else:
        # Per block cast case: 128 * 64
        TILE_M, TILE_K = 128, 64

    num_vectorize = math.gcd(TILE_M * TILE_K // num_threads, get_best_vectorize_size(out_dtype))
    # Keep row and column mappings separable for TileLang's layout inversion.
    num_threads_per_row = math.gcd(num_threads, TILE_K // num_vectorize)
    num_rows = num_threads // num_threads_per_row

    def out_layout_fn(i, j):
        thread = i % num_rows * num_threads_per_row + j // num_vectorize % num_threads_per_row
        index = i // num_rows * (TILE_K // num_threads_per_row) + j // (num_threads_per_row * num_vectorize) * num_vectorize + j % num_vectorize
        return thread, index

    sf_size = T.ceildiv(TILE_M, num_per_tokens) * T.ceildiv(TILE_K, num_per_channels)
    sf_size_aligned = align(sf_size, num_threads)
    num_blocks_per_token = T.ceildiv(hidden, TILE_K)

    # Runtime symbols
    num_tokens = T.dynamic('num_tokens')
    sf_shape = get_sf_shape((num_tokens, hidden), in_config)

    sf_stride = T.dynamic('sf_stride')

    @T.prim_func
    def cast_back_kernel(
        x: T.Tensor[(num_tokens, hidden), in_config.dtype],
        x_sf: T.StridedTensor[sf_shape, (sf_stride, 1), in_config.sf_dtype],
        out: T.Tensor[(num_tokens, hidden), out_dtype],
    ):
        T.assume(num_tokens > 0)
        with T.Kernel(T.ceildiv(num_tokens, TILE_M) * num_blocks_per_token, threads=num_threads) as pid:
            pid_token = pid // num_blocks_per_token
            pid_hidden = pid % num_blocks_per_token
            if use_pdl:
                T.pdl_sync()
            x_shared = T.alloc_shared((TILE_M, TILE_K), in_config.dtype)
            sf_shared = T.alloc_shared((T.ceildiv(TILE_M, num_per_tokens), T.ceildiv(TILE_K, num_per_channels)), T.float32)
            out_fragment = T.alloc_fragment((TILE_M, TILE_K), out_dtype)

            if num_per_channels > 1:
                T.annotate_layout({out_fragment: T.Fragment((TILE_M, TILE_K), forward_fn=out_layout_fn)})

            T.copy(x[pid_token * TILE_M, pid_hidden * TILE_K], x_shared, disable_tma=True)
            for id in T.Parallel(sf_size_aligned):
                if id < sf_size:
                    i, j = id // T.ceildiv(TILE_K, num_per_channels), id % T.ceildiv(TILE_K, num_per_channels)
                    token_index = pid_token * TILE_M // num_per_tokens + i
                    channel_index = pid_hidden * TILE_K // num_per_channels + j
                    sf = load_sf(x_sf, token_index, channel_index, in_config)
                    sf_shared[i, j] = transform_sf(sf, in_config)

            for i, j in T.Parallel(TILE_M, TILE_K):
                out_fragment[i, j] = x_shared[i, j] * sf_shared[i // num_per_tokens, j // num_per_channels]

            T.copy(out_fragment, out[pid_token * TILE_M, pid_hidden * TILE_K])

            if use_pdl:
                T.pdl_trigger()

    return cast_back_kernel

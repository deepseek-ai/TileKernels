import math

import tilelang
from tilelang import language as T

from tile_kernels.utils import ceil_div


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def get_engram_grad_w_reduce_kernel_cuda(
    hidden_size: int,
    num_persistent_blocks: int,
    hc_mult: int = 4,
    has_value_norm: bool = False,
    dtype: T.dtype = T.bfloat16,
    use_pdl: bool = False,
):
    """Reduce grad_w_partial over persistent blocks, fused with weight multiply and accumulation."""
    assert dtype in (T.bfloat16, T.float32)
    assert hidden_size % 128 == 0
    blk_d = math.gcd(256, hidden_size)
    vec_size = 4
    threads = blk_d // vec_size
    num_tiles = hidden_size // blk_d
    num_batches = 4
    num_rows = ceil_div(num_persistent_blocks, num_batches)

    @T.prim_func
    def engram_grad_w_reduce_kernel(
        grad_w_partial: T.Tensor[(num_persistent_blocks, hc_mult + has_value_norm, hidden_size), T.float],
        weight_hidden: T.Tensor[(hc_mult, hidden_size), dtype],
        weight_embed: T.Tensor[(hc_mult, hidden_size), dtype],
        grad_weight_hidden: T.Tensor[(hc_mult, hidden_size), T.float],
        grad_weight_embed: T.Tensor[(hc_mult, hidden_size), T.float],
        grad_weight_value: T.Tensor[(hidden_size,), T.float],
    ):
        with T.Kernel(hc_mult + has_value_norm, num_tiles, threads=threads) as (pid_h, pid_b):
            if use_pdl:
                T.pdl_sync()
            wh_fragment = T.alloc_fragment((blk_d,), T.float)
            we_fragment = T.alloc_fragment((blk_d,), T.float)
            grad_w_shared = T.alloc_shared((num_rows, blk_d), T.float)
            grad_w_fragment = T.alloc_fragment((blk_d,), T.float)
            grad_wh_fragment = T.alloc_fragment((blk_d,), T.float)
            grad_we_fragment = T.alloc_fragment((blk_d,), T.float)

            T.clear(grad_w_fragment)
            if pid_h < hc_mult:
                T.copy(weight_hidden[pid_h, pid_b * blk_d : (pid_b + 1) * blk_d], wh_fragment, disable_tma=True)
                T.copy(weight_embed[pid_h, pid_b * blk_d : (pid_b + 1) * blk_d], we_fragment, disable_tma=True)
                T.copy(grad_weight_hidden[pid_h, pid_b * blk_d : (pid_b + 1) * blk_d], grad_wh_fragment, disable_tma=True)
                T.copy(grad_weight_embed[pid_h, pid_b * blk_d : (pid_b + 1) * blk_d], grad_we_fragment, disable_tma=True)

            for i_r in T.Pipelined(0, num_batches, num_stages=2):
                T.copy(
                    grad_w_partial[i_r * num_rows : (i_r + 1) * num_rows, pid_h, pid_b * blk_d : (pid_b + 1) * blk_d], grad_w_shared, disable_tma=True
                )

                for i in T.Serial(num_rows):
                    for j in T.Parallel(blk_d):
                        grad_w_fragment[j] += grad_w_shared[i, j]

            if pid_h < hc_mult:
                for i in T.Parallel(blk_d):
                    grad_wh_fragment[i] += grad_w_fragment[i] * we_fragment[i]
                    grad_we_fragment[i] += grad_w_fragment[i] * wh_fragment[i]
                T.copy(grad_we_fragment, grad_weight_embed[pid_h, pid_b * blk_d : (pid_b + 1) * blk_d], disable_tma=True)
                T.copy(grad_wh_fragment, grad_weight_hidden[pid_h, pid_b * blk_d : (pid_b + 1) * blk_d], disable_tma=True)
            else:
                for i in T.Parallel(blk_d):
                    grad_weight_value[pid_b * blk_d + i] += grad_w_fragment[i]
            if use_pdl:
                T.pdl_trigger()

    return engram_grad_w_reduce_kernel

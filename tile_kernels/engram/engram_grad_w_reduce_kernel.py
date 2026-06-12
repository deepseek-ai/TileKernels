import os

import torch
import tilelang
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def get_engram_grad_w_reduce_kernel(
    hidden_size: int,
    num_persistent_blocks: int,
    hc_mult: int = 4,
):
    """Reduce grad_w_partial over persistent blocks, fused with weight multiply and accumulation."""
    threads = 128
    blk_d = 512
    assert hidden_size % blk_d == 0
    num_tiles = hidden_size // blk_d
    # SM90/SM100 devices can have SM counts such as 114, so avoid assuming
    # that the persistent-block count is always divisible by four.
    num_batches = max(batch for batch in range(4, 0, -1) if num_persistent_blocks % batch == 0)
    num_rows = num_persistent_blocks // num_batches

    @T.prim_func
    def engram_grad_w_reduce_kernel(
        grad_w_partial: T.Tensor[(num_persistent_blocks, hc_mult, hidden_size), T.float],
        weight_hidden: T.Tensor[(hc_mult, hidden_size), T.bfloat16],
        weight_embed: T.Tensor[(hc_mult, hidden_size), T.bfloat16],
        grad_weight_hidden: T.Tensor[(hc_mult, hidden_size), T.float],
        grad_weight_embed: T.Tensor[(hc_mult, hidden_size), T.float],
    ):
        with T.Kernel(hc_mult, num_tiles, threads=threads) as (pid_h, pid_b):
            wh_fragment = T.alloc_fragment((blk_d,), T.float)
            we_fragment = T.alloc_fragment((blk_d,), T.float)
            grad_w_shared = T.alloc_shared((num_rows, blk_d), T.float)
            grad_w_fragment = T.alloc_fragment((blk_d,), T.float)
            grad_wh_fragment = T.alloc_fragment((blk_d,), T.float)
            grad_we_fragment = T.alloc_fragment((blk_d,), T.float)

            T.clear(grad_w_fragment)
            T.copy(weight_hidden[pid_h, pid_b * blk_d : (pid_b + 1) * blk_d], wh_fragment)
            T.copy(weight_embed[pid_h, pid_b * blk_d : (pid_b + 1) * blk_d], we_fragment)
            T.copy(grad_weight_hidden[pid_h, pid_b * blk_d : (pid_b + 1) * blk_d], grad_wh_fragment)
            T.copy(grad_weight_embed[pid_h, pid_b * blk_d : (pid_b + 1) * blk_d], grad_we_fragment)

            for i_r in T.Pipelined(0, num_batches, num_stages=2):
                T.copy(grad_w_partial[i_r * num_rows : (i_r + 1) * num_rows, pid_h, pid_b * blk_d : (pid_b + 1) * blk_d], grad_w_shared)

                for i in T.Serial(num_rows):
                    for j in T.Parallel(blk_d):
                        grad_w_fragment[j] += grad_w_shared[i, j]

            for j in T.Parallel(blk_d):
                grad_wh_fragment[j] += grad_w_fragment[j] * we_fragment[j]
                grad_we_fragment[j] += grad_w_fragment[j] * wh_fragment[j]

            T.copy(grad_we_fragment, grad_weight_embed[pid_h, pid_b * blk_d : (pid_b + 1) * blk_d])
            T.copy(grad_wh_fragment, grad_weight_hidden[pid_h, pid_b * blk_d : (pid_b + 1) * blk_d])

    return engram_grad_w_reduce_kernel


def grad_w_reduce(
    grad_w_partial: torch.Tensor,
    weight_hidden: torch.Tensor,
    weight_embed: torch.Tensor,
    grad_weight_hidden: torch.Tensor,
    grad_weight_embed: torch.Tensor,
) -> None:
    """Reduce grad_w_partial over persistent blocks, fused with weight multiply, accumulating into grad_weight tensors.

    Args:
        grad_w_partial: Partial weight gradients, shape (num_persistent_blocks, hc_mult, hidden_size), float32.
        weight_hidden: RMSNorm weight for hidden states, shape (hc_mult, hidden_size), bfloat16.
        weight_embed: RMSNorm weight for key embeddings, shape (hc_mult, hidden_size), bfloat16.
        grad_weight_hidden: Accumulated gradient for weight_hidden, shape (hc_mult, hidden_size), float32. Modified in-place.
        grad_weight_embed: Accumulated gradient for weight_embed, shape (hc_mult, hidden_size), float32. Modified in-place.
    """
    num_persistent_blocks, hc_mult, hidden_size = grad_w_partial.shape

    kernel = get_engram_grad_w_reduce_kernel(hidden_size, num_persistent_blocks, hc_mult)
    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    kernel(grad_w_partial, weight_hidden, weight_embed, grad_weight_hidden, grad_weight_embed)

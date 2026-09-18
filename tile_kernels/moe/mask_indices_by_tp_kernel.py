import os
import torch
import tilelang
from tilelang import language as T

from tile_kernels.utils import get_device_guard


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_mask_indices_by_tp_kernel(num_topk: int, dtype: T.dtype):
    num_threads = 128

    num_tokens = T.dynamic('num_tokens')
    num_blocks = T.ceildiv(num_tokens * num_topk, num_threads)

    @T.prim_func
    def mask_indices_by_tp_kernel(
        indices: T.Tensor[(num_tokens, num_topk), dtype],
        masked_indices: T.Tensor[(num_tokens, num_topk), dtype],
        per_gpu: T.int32,
        per_dp: T.int32,
        num_tp_ranks: T.int32,
        tp_rank: T.int32,
    ):
        with T.Kernel(num_blocks, threads=num_threads) as (pid, ):
            indices_1d = T.reshape(indices, (num_tokens * num_topk, ))
            masked_indices_1d = T.reshape(masked_indices, (num_tokens * num_topk, ))
            thread_idx = T.get_thread_binding()
            index = pid * num_threads + thread_idx

            value = T.alloc_var(dtype)
            if index < num_tokens * num_topk:
                value = indices_1d[index]
                if value < 0 or T.truncmod(T.truncdiv(value, per_gpu), num_tp_ranks) != tp_rank:
                    masked_indices_1d[index] = -1
                else:
                    value -= tp_rank * per_gpu
                    dp_rank = T.truncdiv(value, per_dp)
                    value -= dp_rank * (per_dp - per_gpu)
                    masked_indices_1d[index] = T.Select(value < 0, T.int64(-1), value)

    return mask_indices_by_tp_kernel


def mask_indices_by_tp(indices: torch.Tensor, n: int, num_ep_ranks: int, tp_rank: int, num_tp_ranks: int) -> torch.Tensor:
    """Mask expert indices to keep only those belonging to the given TP rank.

    Args:
        indices: Expert index tensor of shape (num_tokens, num_topk).
        n: Total number of experts across all ranks.
        num_ep_ranks: Expert-parallelism size.
        tp_rank: Tensor-parallelism rank of the current device.
        num_tp_ranks: Tensor-parallelism size.

    Returns:
        Masked index tensor with non-local experts set to -1 and local
        indices remapped to the local expert range.
    """
    num_topk = indices.shape[1]
    per_gpu = n // num_ep_ranks
    per_dp = num_tp_ranks * per_gpu
    # Compile and launch on the device of the input tensor, which is not
    # necessarily the current CUDA device.
    with get_device_guard(indices.device):
        kernel = get_mask_indices_by_tp_kernel(num_topk, T.dtype(indices.dtype))

        if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
            print(kernel.get_kernel_source())

        masked_indices = torch.empty_like(indices)
        if indices.shape[0] > 0:
            kernel(indices, masked_indices, per_gpu, per_dp, num_tp_ranks, tp_rank)

    return masked_indices

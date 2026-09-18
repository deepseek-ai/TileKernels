import os
import torch
import tilelang
from tilelang import language as T

from tile_kernels.config import get_num_sms
from tile_kernels.utils import align, get_device_guard


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_group_count_kernel(num_topk: int, num_groups: int, num_sms: int):
    num_threads = 128
    num_blocks = num_sms * 2
    num_tokens = T.dynamic('num_tokens')

    @T.prim_func
    def group_count_kernel(
        group_idx: T.Tensor[(num_tokens, num_topk), T.int64],
        out: T.Tensor[(num_groups, ), T.int32],
    ):
        with T.Kernel(num_blocks, threads=num_threads) as (pid, ):
            thread_idx = T.get_thread_binding()
            global_thread_idx = pid * num_threads + thread_idx

            out_shared = T.alloc_shared((align(num_groups, num_threads), ), T.int32)
            T.clear(out_shared)
            T.sync_threads()

            for i in T.serial(global_thread_idx, num_tokens, num_blocks * num_threads):
                for j in T.unroll(num_topk):
                    expert_idx = T.int32(group_idx[i, j])
                    T.device_assert(-1 <= expert_idx < num_groups)
                    T.assume(expert_idx < num_groups)
                    if expert_idx >= 0:
                        T.atomic_add(out_shared[expert_idx], 1)

            T.sync_threads()
            for i in T.serial(thread_idx, num_groups, num_threads):
                if out_shared[i] > 0:
                    T.atomic_add(out[i], out_shared[i])

    return group_count_kernel


def group_count(group_idx: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Count the number of tokens assigned to each expert.

    Args:
        group_idx: Int64 expert index tensor of shape (num_tokens, num_topk).
        num_groups: Total number of experts.

    Returns:
        Int32 tensor of shape (num_groups,) with per-expert token counts.
    """
    assert group_idx.dim() == 2 and group_idx.is_contiguous()

    # Allocate and launch on the device of the input tensor, which is not
    # necessarily the current CUDA device.
    device = group_idx.device
    with get_device_guard(device):
        kernel = get_group_count_kernel(group_idx.shape[1], num_groups, get_num_sms(device.index))

        if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
            print(kernel.get_kernel_source())

        out = torch.zeros(num_groups, dtype=torch.int32, device=device)
        kernel(group_idx, out)

    return out

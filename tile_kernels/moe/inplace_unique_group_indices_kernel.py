import os
import torch
import tilelang
from tilelang import language as T

from tile_kernels.config import get_num_sms
from tile_kernels.utils import align, get_device_guard


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True,
    }
)
def get_inplace_unique_group_indices_kernel(num_topk: int, num_groups_aligned: int, num_sms: int):
    num_threads = 128
    num_tokens = T.dynamic('num_tokens')

    grid_x = num_sms * 2

    @T.prim_func
    def inplace_unique_group_indices_kernel(
        group_indices: T.Tensor[(num_tokens, num_topk), T.int64],
    ):
        with T.Kernel(grid_x, threads=num_threads) as (pid_token, ):
            thread_idx = T.get_thread_binding()
            global_thread_idx = pid_token * num_threads + thread_idx

            group_sel = T.alloc_local((2, ), T.uint64)

            for i in T.serial(global_thread_idx, num_tokens, grid_x * num_threads):
                for j in T.unroll(num_groups_aligned // 64):
                    group_sel[j] = 0
                for j in T.unroll(num_topk):
                    group_idx = group_indices[i, j]
                    T.device_assert(group_idx < num_groups_aligned)
                    T.assume(group_idx < num_groups_aligned)
                    mask = T.Select(group_idx >= 0, T.uint64(1) << (group_idx % 64), T.uint64(0))
                    lo_mask = T.Select(group_idx < 64, mask, T.uint64(0))
                    hi_mask = T.Select(group_idx >= 64, mask, T.uint64(0))
                    found = (lo_mask & group_sel[0]) | (hi_mask & group_sel[1])
                    group_sel[0] |= lo_mask
                    group_sel[1] |= hi_mask
                    if found:
                        group_indices[i, j] = -1

    return inplace_unique_group_indices_kernel


def inplace_unique_group_indices(group_indices: torch.Tensor, num_groups: int) -> None:
    """Deduplicate group indices per token, marking duplicates as -1 in-place.

    Args:
        group_indices: Int64 tensor of shape (num_tokens, num_topk) with group ids.
        num_groups: Total number of groups (must be <= 128).
    """
    assert group_indices.dim() == 2
    assert num_groups <= 128

    num_topk = group_indices.shape[1]
    num_groups_aligned = align(num_groups, 64)

    # Compile and launch on the device of the input tensor, which is not
    # necessarily the current CUDA device.
    device = group_indices.device
    with get_device_guard(device):
        kernel = get_inplace_unique_group_indices_kernel(num_topk, num_groups_aligned, get_num_sms(device.index))

        if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
            print(kernel.get_kernel_source())

        if group_indices.shape[0] > 0:
            kernel(group_indices)

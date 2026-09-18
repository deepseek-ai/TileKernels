import os
import torch
import tilelang
from tilelang import language as T
from tile_kernels.utils import align, get_device_guard
from tile_kernels.config import get_num_sms


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_aux_fi_kernel(num_topk: int, num_experts: int, num_sms: int):
    num_threads = 128
    num_tokens = T.dynamic('num_tokens')

    num_blocks = num_sms * 2

    @T.prim_func
    def aux_fi_kernel(
        topk_idx: T.Tensor[(num_tokens, num_topk), T.int64],
        out: T.Tensor[(num_experts, ), T.float32],
        num_aux_topk: T.int32,
    ):
        with T.Kernel(num_blocks, threads=num_threads) as (pid, ):
            thread_idx = T.get_thread_binding()
            global_thread_idx = pid * num_threads + thread_idx

            out_shared = T.alloc_shared((align(num_experts, num_threads), ), T.int32)
            T.clear(out_shared)
            T.sync_threads()

            for i in T.serial(global_thread_idx, num_tokens, num_blocks * num_threads):
                for j in T.unroll(num_topk):
                    expert_idx = T.int32(topk_idx[i, j])
                    T.device_assert(-1 <= expert_idx < num_experts)
                    T.assume(expert_idx < num_experts)
                    if expert_idx >= 0:
                        T.atomic_add(out_shared[expert_idx], 1)

            T.sync_threads()
            for i in T.serial(thread_idx, num_experts, num_threads):
                if out_shared[i] > 0:
                    T.atomic_add(out[i], T.cast(out_shared[i] * num_experts, T.float32) / T.cast(num_tokens * num_aux_topk, T.float32))

    return aux_fi_kernel


def aux_fi(topk_idx: torch.Tensor, num_experts: int, num_aux_topk: int) -> torch.Tensor:
    """Compute per-expert frequency indicator (f_i) for the auxiliary loss.

    Args:
        topk_idx: Int64 expert index tensor of shape (num_tokens, num_topk).
        num_experts: Total number of experts.
        num_aux_topk: Number of top-k selections used in the auxiliary loss.

    Returns:
        FP32 tensor of shape (num_experts,) with frequency indicators.
    """
    assert topk_idx.dim() == 2 and topk_idx.is_contiguous()

    num_topk = topk_idx.shape[1]

    # Allocate and launch on the device of the input tensor, which is not
    # necessarily the current CUDA device.
    device = topk_idx.device
    with get_device_guard(device):
        kernel = get_aux_fi_kernel(num_topk, num_experts, get_num_sms(device.index))

        if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
            print(kernel.get_kernel_source())

        out = torch.zeros(num_experts, dtype=torch.float32, device=device)
        kernel(topk_idx, out, num_aux_topk)

    return out

import os
import torch
import tilelang
from tilelang import language as T
from tilelang.utils.target import determine_target


def _is_hip() -> bool:
    return determine_target(return_object=True).kind.name == 'hip'


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_normalize_weight_kernel(num_topk: int):
    num_threads = 128
    # T.vectorized generates vector load/store instructions on CUDA (e.g. float4),
    # but produces NaN outputs on HIP due to AMD backend codegen limitations.
    loop = T.unroll if _is_hip() else T.vectorized

    num_tokens = T.dynamic('num_tokens')
    num_blocks = T.ceildiv(num_tokens, 128)

    @T.prim_func
    def normalize_weight_kernel(
        topk_weights: T.Tensor[(num_tokens, num_topk), T.float32],
        denominator: T.Tensor[(num_tokens,), T.float32],
        normalized_weights: T.Tensor[(num_tokens, num_topk), T.float32],
    ):
        with T.Kernel(num_blocks, threads=num_threads) as (pid, ):
            tid = T.get_thread_binding()
            weights_local = T.alloc_local((num_topk,), T.float32)
            row = pid * num_threads + tid

            if row < num_tokens:
                # NOTE: Align with top2_sum_gate kernel implementation
                # Use T.alloc_local + explicit BufferStore for initialization.
                # T.alloc_var(init=float_literal) uses block_attr which is not
                # reliably lowered to initialization code on all backends (e.g.
                # the generated HIP kernel omits the assignment, leaving the
                # register uninitialized and producing NaN on AMD hardware).
                sum = T.alloc_local((1,), T.float32)
                sum[0] = 1e-20
                for i in loop(num_topk):
                    weights_local[i] = topk_weights[row, i]

                for i in T.unroll(num_topk):
                    sum[0] = sum[0] + weights_local[i]

                denominator[row] = sum[0]
                for i in loop(num_topk):
                    normalized_weights[row, i] = weights_local[i] / sum[0]

    return normalize_weight_kernel


def normalize_weight(topk_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize top-k routing weights so that each token's weights sum to one.

    Args:
        topk_weights: FP32 tensor of shape (num_tokens, num_topk).

    Returns:
        A tuple ``(denominator, normalized_weights)`` where ``denominator`` has
        shape (num_tokens,) and ``normalized_weights`` has shape (num_tokens, num_topk).
    """
    assert topk_weights.dim() == 2 and topk_weights.is_contiguous()
    assert topk_weights.dtype == torch.float32

    num_tokens, num_topk = topk_weights.shape
    kernel = get_normalize_weight_kernel(num_topk)

    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    denominator = torch.empty((num_tokens,), dtype=torch.float32, device='cuda')
    normalized_weights = torch.empty((num_tokens, num_topk), dtype=torch.float32, device='cuda')

    if num_tokens > 0:
        kernel(topk_weights, denominator, normalized_weights)

    return (denominator, normalized_weights)

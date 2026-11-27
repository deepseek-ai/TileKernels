import math
import tilelang
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_engram_fused_weight_kernel_cuda(
    hidden_size: int,
    hc_mult: int,
    dtype: T.dtype = T.bfloat16,
    use_pdl: bool = False,
):
    """Elementwise bf16/fp32 weights -> fp32 for weight_hidden * weight_embed."""
    assert dtype in (T.bfloat16, T.float32)
    threads = 32
    vec_size = math.gcd(8, hidden_size // 32)
    blk_d = threads * vec_size
    assert hidden_size % blk_d == 0
    num_blk = hidden_size // blk_d

    @T.prim_func
    def engram_fused_weight_kernel_cuda(
        weight_hidden: T.Tensor[(hc_mult, hidden_size), dtype],
        weight_embed: T.Tensor[(hc_mult, hidden_size), dtype],
        weight_fused: T.Tensor[(hc_mult, hidden_size), T.float],
    ):
        with T.Kernel(hc_mult, num_blk, threads=threads) as (pid_h, pid_b):
            if use_pdl:
                T.pdl_sync()
            tid = T.get_thread_binding()
            a_local = T.alloc_local((vec_size,), T.float)
            b_local = T.alloc_local((vec_size,), T.float)
            for i_k in T.vectorized(vec_size):
                a_local[i_k] = weight_hidden[pid_h, pid_b * blk_d + tid * vec_size + i_k]
                b_local[i_k] = weight_embed[pid_h, pid_b * blk_d + tid * vec_size + i_k]
            for i_k in T.vectorized(vec_size):
                weight_fused[pid_h, pid_b * blk_d + tid * vec_size + i_k] = a_local[i_k] * b_local[i_k]
            if use_pdl:
                T.pdl_trigger()

    return engram_fused_weight_kernel_cuda

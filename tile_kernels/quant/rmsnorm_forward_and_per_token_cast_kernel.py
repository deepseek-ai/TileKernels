import os

import torch
import tilelang
from tilelang import language as T

from tile_kernels.quant.common import *


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_rmsnorm_forward_and_per_token_cast_kernel(
    hidden: int,
    eps: float,
    in_dtype: T.dtype,
    out_config: CastOutputConfig,
):
    num_threads = 128
    num_per_channels = out_config.sf_block[1]
    num_groups = hidden // num_per_channels

    num_tokens = T.dynamic('num_tokens')
    sf_shape = get_sf_shape((num_tokens, hidden), out_config)
    sf_stride = T.dynamic('sf_stride')

    @T.prim_func
    def rmsnorm_forward_and_per_token_cast_kernel(
        x: T.Tensor[(num_tokens, hidden), in_dtype],
        weight: T.Tensor[(hidden,), in_dtype],
        out: T.Tensor[(num_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[sf_shape, (sf_stride, 1), out_config.sf_dtype],
    ):
        with T.Kernel(num_tokens, threads=num_threads) as pid:
            tid = T.get_thread_binding()
            warp_partial = T.alloc_shared((num_threads // 32,), T.float32)
            rstd_shared = T.alloc_shared((1,), T.float32)
            sf_inv_shared = T.alloc_shared((1,), T.float32)
            sum_square = T.alloc_var(T.float32, init=0.0)

            for j in T.serial(hidden // num_threads):
                value = T.float32(x[pid, tid + j * num_threads])
                sum_square += value * value

            sum_square = T.warp_reduce_sum(sum_square)
            if tid % 32 == 0:
                warp_partial[tid // 32] = sum_square
            T.sync_threads()

            if tid < 32:
                sum_square = T.Select(tid < num_threads // 32, warp_partial[tid], 0.0)
                sum_square = T.warp_reduce_sum(sum_square)
                if tid == 0:
                    rstd_shared[0] = T.rsqrt(sum_square / hidden + eps)
            T.sync_threads()

            if num_per_channels == 128:
                for group_batch in T.serial(T.ceildiv(num_groups, num_threads // 32)):
                    group = group_batch * (num_threads // 32) + tid // 32
                    if group < num_groups:
                        values = T.alloc_local((4,), T.float32)
                        amax = T.alloc_var(T.float32, init=0.0)
                        for j in T.serial(4):
                            offset = group * num_per_channels + tid % 32 + j * 32
                            values[j] = T.float32(x[pid, offset]) * T.float32(weight[offset]) * rstd_shared[0]
                            amax = T.max(amax, T.abs(values[j]))

                        amax = T.warp_reduce_max(amax)
                        sf_inv_local = T.alloc_var(T.float32)
                        if tid % 32 == 0:
                            sf, sf_inv = get_sf_and_inv(amax, out_config)
                            store_sf(out_sf, sf, pid, group, out_config)
                            sf_inv_local = sf_inv
                        sf_inv_local = T.shfl_sync(sf_inv_local, 0)

                        for j in T.serial(4):
                            offset = group * num_per_channels + tid % 32 + j * 32
                            out[pid, offset] = values[j] * sf_inv_local
            else:
                amax = T.alloc_var(T.float32, init=0.0)
                for j in T.serial(hidden // num_threads):
                    offset = tid + j * num_threads
                    value = T.float32(x[pid, offset]) * T.float32(weight[offset]) * rstd_shared[0]
                    amax = T.max(amax, T.abs(value))

                amax = T.warp_reduce_max(amax)
                if tid % 32 == 0:
                    warp_partial[tid // 32] = amax
                T.sync_threads()

                if tid < 32:
                    block_amax = T.alloc_var(T.float32)
                    block_amax = T.Select(tid < num_threads // 32, warp_partial[tid], 0.0)
                    block_amax = T.warp_reduce_max(block_amax)
                    if tid == 0:
                        sf, sf_inv = get_sf_and_inv(block_amax, out_config)
                        store_sf(out_sf, sf, pid, 0, out_config)
                        sf_inv_shared[0] = sf_inv
                T.sync_threads()

                for j in T.serial(hidden // num_threads):
                    offset = tid + j * num_threads
                    value = T.float32(x[pid, offset]) * T.float32(weight[offset]) * rstd_shared[0]
                    out[pid, offset] = value * sf_inv_shared[0]

    return rmsnorm_forward_and_per_token_cast_kernel


def rmsnorm_forward_and_per_token_cast(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    fmt: str,
    num_per_channels: int,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> QuantTensor:
    """Fuse RMSNorm forward with per-token FP8 quantization.

    Args:
        x: Contiguous BF16 or FP32 input of shape ``(num_tokens, hidden)``.
        weight: Contiguous RMSNorm weight of shape ``(hidden,)`` and the same dtype as ``x``.
        eps: Epsilon for RMSNorm numerical stability.
        fmt: Target format (must be ``'e4m3'``).
        num_per_channels: Number of channels in each scaling block (128 or ``hidden``).
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major sf factors.
        round_sf: Whether to round scaling factors to powers of two.
        use_packed_ue8m0: Whether to use packed UE8M0 format for sf factors.

    Returns:
        A tuple ``(out, out_sf)`` with FP8 output and sf-factor tensor.
    """
    assert x.dim() == 2 and x.is_contiguous()
    assert x.dtype in (torch.bfloat16, torch.float32)
    assert weight.dim() == 1 and weight.is_contiguous()
    assert weight.dtype == x.dtype

    num_tokens, hidden = x.shape
    assert weight.shape[0] == hidden
    assert hidden % 128 == 0
    assert num_per_channels in (128, hidden)
    assert num_per_channels == 128 or not use_tma_aligned_col_major_sf
    assert fmt == 'e4m3'

    out_config = get_cast_output_config(
        fmt,
        (1, num_per_channels),
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
    )
    kernel = get_rmsnorm_forward_and_per_token_cast_kernel(
        hidden=hidden,
        eps=eps,
        in_dtype=T.dtype(x.dtype),
        out_config=out_config,
    )

    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    out = torch.empty((num_tokens, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    out_sf = alloc_scaling_factors((num_tokens, hidden), out_config, device=x.device)
    if num_tokens > 0:
        kernel(x, weight, out, out_sf)

    out_sf = cast_epilogue(out_sf, num_tokens, hidden, out_config)
    return out, out_sf

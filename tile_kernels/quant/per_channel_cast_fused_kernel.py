import os
from typing import Optional

import tilelang
import torch
from tilelang import language as T

from tile_kernels.quant.common import *
from tile_kernels.utils import ceil_div


def transform_token_idx(with_expand: bool, idx: int, token_idx: int, x):
    if with_expand:
        return x[idx]
    return token_idx


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_per_channel_cast_fused_kernel(
    hidden: int,
    with_expand: bool,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
):
    num_tokens = T.dynamic('num_tokens')
    num_tokens_out = T.dynamic('num_tokens_out')

    num_per_tokens, _ = out_config.sf_block
    _, num_per_channels = in_config.sf_block
    assert num_per_tokens == 128
    assert not in_config.with_sf or num_per_channels == 128

    num_threads = 256
    TILE_M, TILE_K = 128, 128
    if in_config.with_sf:
        TILE_K = 256

    # Set num_threads_per_token = 32 to avoid bank conflict
    num_threads_per_token = 32
    assert TILE_K % num_threads_per_token == 0

    # Each thread processes a block of size VEC_M * VEC_K
    VEC_K = TILE_K // num_threads_per_token
    VEC_M = TILE_M * num_threads_per_token // num_threads

    @T.prim_func
    def per_channel_cast_fused_kernel(
        x: T.Tensor[(num_tokens, hidden), in_config.dtype],
        out: T.Tensor[(num_tokens_out, hidden), out_config.dtype],
        out_sf: T.Tensor[(T.ceildiv(num_tokens_out, num_per_tokens), hidden), out_config.sf_dtype],
        x_sf_invs: T.Tensor[(num_tokens, T.ceildiv(hidden, num_per_channels)), in_config.sf_dtype],
        pos_to_token: T.Tensor[(num_tokens_out,), T.int32],
    ):
        with T.Kernel(T.ceildiv(num_tokens_out, TILE_M), T.ceildiv(hidden, TILE_K), threads=num_threads) as (pid_token, pid_hidden):
            x_shared = T.alloc_shared((TILE_M, TILE_K), in_config.dtype)
            pos_to_token_local = T.alloc_local((VEC_M,), T.int32)
            sf_invs_local = T.alloc_local((VEC_M,), T.float32)
            amax_local = T.alloc_local((VEC_K,), T.float32)
            amax_shared = T.alloc_shared((VEC_K, num_threads), T.float32)
            in_local = T.alloc_local((VEC_K,), in_config.dtype)
            out_local = T.alloc_local((VEC_K,), out_config.dtype)
            tid = T.get_thread_binding(0)
            m_id, k_id = tid // num_threads_per_token, tid % num_threads_per_token
            m_offset = pid_token * TILE_M + m_id * VEC_M
            k_offset = pid_hidden * TILE_K + k_id * VEC_K

            T.assume(num_tokens_out % 128 == 0 or (with_expand and num_tokens_out % 16 == 0))
            if with_expand:
                tmp = T.alloc_var(T.int32)
                # Invalid tail rows remain -1 and follow the existing zero-fill path,
                # so they do not participate in the amax reduction.
                tmp = -1
                row = k_id + m_offset
                if k_id < VEC_M and row < num_tokens_out:
                    tmp = pos_to_token[row]

                for i in T.serial(VEC_M):
                    pos_to_token_local[i] = T.shfl_sync(tmp, i)

            if in_config.with_sf:
                for i in T.serial(VEC_M):
                    pos = transform_token_idx(with_expand, i, i + m_offset, pos_to_token_local)
                    T.assume(pos < num_tokens)
                    sf_invs_local[i] = T.Select(with_expand and pos < 0, 0.0, x_sf_invs[pos, (pid_hidden * TILE_K + k_id * VEC_K) // num_per_channels])

            T.clear(amax_local)
            for i in T.serial(VEC_M):
                pos = transform_token_idx(with_expand, i, i + m_offset, pos_to_token_local)
                T.assume(pos < num_tokens)
                if not with_expand or pos >= 0:
                    for j in T.vectorized(VEC_K):
                        T.assume(pos < num_tokens)
                        in_local[j] = x[pos, j + k_offset]
                        x_shared[i + m_id * VEC_M, j + k_id * VEC_K] = in_local[j]

                    for j in T.vectorized(VEC_K):
                        if in_config.with_sf:
                            amax_local[j] = T.max(amax_local[j], T.abs(in_local[j] * sf_invs_local[i]))
                        else:
                            amax_local[j] = T.max(amax_local[j], T.abs(in_local[j]))
                else:
                    for j in T.vectorized(VEC_K):
                        x_shared[i + m_id * VEC_M, j + k_id * VEC_K] = 0

            for i in T.unroll(VEC_K):
                amax_shared[i, tid] = amax_local[i]

            sf = T.alloc_var(T.float32)
            sf = 0
            col_id = tid % num_threads_per_token * VEC_K + tid // num_threads_per_token
            if tid < TILE_K:
                for i in T.serial(col_id // VEC_K, num_threads, num_threads_per_token):
                    sf = T.max(sf, amax_shared[col_id % VEC_K, i])

                sf, sf_inv = get_sf_and_inv(sf, out_config)
                out_sf[pid_token, pid_hidden * TILE_K + col_id] = sf
                amax_shared[0, tid] = sf_inv

            # Reuse amax_local as sf
            for i in T.serial(VEC_K):
                amax_local[i] = amax_shared[0, k_id + i * num_threads_per_token]

            for i in T.serial(VEC_M):
                for j in T.vectorized(VEC_K):
                    in_local[j] = x_shared[i + m_id * VEC_M, j + k_id * VEC_K]
                for j in T.vectorized(VEC_K):
                    if in_config.with_sf:
                        out_local[j] = in_local[j] * sf_invs_local[i] * amax_local[j]
                    else:
                        out_local[j] = in_local[j] * amax_local[j]
                for j in T.vectorized(VEC_K):
                    out[i + m_offset, j + k_offset] = out_local[j]

    return per_channel_cast_fused_kernel


def per_channel_cast_fused(
    x: Union[torch.Tensor, QuantTensor],
    fmt: str,
    num_per_tokens: int,
    round_sf: bool = False,
    num_per_channels: Optional[int] = None,
    pos_to_token: Optional[torch.Tensor] = None,
) -> QuantTensor:
    """Cast a matrix to FP8 with per-channel scaling, optionally fusing resf and token expansion.

    Args:
        x: Input tensor of shape (num_tokens, hidden), either a plain tensor
            or a ``QuantTensor`` ``(data, sf_invs)`` for rescaling FP8 inputs.
        fmt: Target FP8 format (must be ``'e4m3'``).
        num_per_tokens: Number of tokens in each scaling block.
        round_sf: Whether to round scaling factors to powers of two.
        num_per_channels: Number of channels in each input scaling block.
        pos_to_token: Optional int32 index tensor for token expansion/gather.

    Returns:
        A tuple ``(out, out_sf)`` with FP8 output and sf-factor tensor.
    """

    x, x_sf_invs, in_config = get_cast_input_and_config(
        x,
        None if num_per_channels is None else (1, num_per_channels),
    )

    assert fmt == 'e4m3'
    assert x.dim() == 2 and x.is_contiguous()
    num_tokens, hidden = x.shape
    num_tokens_out = num_tokens

    if pos_to_token is not None:
        assert pos_to_token.dim() == 1 and pos_to_token.is_contiguous()
        assert pos_to_token.dtype == torch.int32
        num_tokens_out = pos_to_token.size(0)
        # Alignment requirement for expanded tokens
        assert num_tokens_out % 16 == 0
    else:
        assert num_tokens_out % 128 == 0

    assert num_per_tokens == 128
    if x_sf_invs is not None:
        assert num_per_channels == 128
        assert x.dtype == torch.float8_e4m3fn
        assert x_sf_invs.dim() == 2 and x_sf_invs.is_contiguous()
        assert x_sf_invs.size(0) == num_tokens and x_sf_invs.size(1) * 128 == hidden


    out_config = get_cast_output_config(fmt, (num_per_tokens, 1), round_sf=round_sf)
    kernel = get_per_channel_cast_fused_kernel(
        hidden,
        with_expand=(pos_to_token is not None),
        in_config=in_config,
        out_config=out_config,
    )

    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    out = torch.empty((num_tokens_out, hidden), dtype=out_config.torch_dtype, device='cuda')
    out_sf = torch.empty((ceil_div(num_tokens_out, num_per_tokens), hidden), dtype=torch.float32, device='cuda')
    if num_tokens_out > 0:
        kernel(x, out, out_sf, x_sf_invs, pos_to_token)

    return out, out_sf

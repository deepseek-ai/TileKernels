import os
import torch
import tilelang
from tilelang import language as T
from typing import Optional, Union
from tile_kernels.quant.common import *
from tile_kernels.utils import get_device_guard


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_reduce_fused_kernel(
    hidden: int,
    num_topk: int,
    in_dtype: T.dtype,
    out_dtype: T.dtype,
    with_sf: bool,
    with_weights: bool,
    with_x_sf: bool,
):
    num_threads = 128

    num_tokens = T.dynamic('num_tokens')
    num_expanded_tokens = T.dynamic('num_expanded_tokens')

    @T.prim_func
    def reduce_fused_kernel(
        x: T.Tensor[(num_expanded_tokens, hidden), in_dtype],
        topk_weights: T.Tensor[(num_tokens, num_topk), T.float32],
        token_topk_to_pos: T.Tensor[(num_tokens, num_topk), T.int32],
        out: T.Tensor[(num_tokens, hidden), out_dtype],
        sf: T.Tensor[(1,), T.float32],
        x_sf: T.Tensor[(num_expanded_tokens,), T.float32],
    ):
        with T.Kernel(num_tokens, threads=num_threads) as (pid_token,):
            reduced_fragment = T.alloc_fragment((hidden,), T.float32)
            topk_weights_local = T.alloc_fragment((num_topk,), T.float32)
            topk_to_pos_local = T.alloc_fragment((num_topk,), T.int32)
            sf_var = T.alloc_var(T.float32)

            T.clear(reduced_fragment)
            if with_sf:
                sf_var = sf[0]
            if with_weights:
                T.copy(topk_weights[pid_token, :], topk_weights_local)
            T.copy(token_topk_to_pos[pid_token, :], topk_to_pos_local)

            for k in T.unroll(num_topk):
                pos = topk_to_pos_local[k]
                T.assume(pos < num_expanded_tokens)
                if pos >= 0:
                    s = T.alloc_var(T.float32)
                    s = 1
                    if with_weights:
                        s = topk_weights_local[k]

                    if with_x_sf:
                        s *= x_sf[pos]
                    for i in T.Parallel(hidden):
                        reduced_fragment[i] += x[pos, i] * s

            for i in T.Parallel(hidden):
                out[pid_token, i] = T.Select(with_sf, reduced_fragment[i] * sf_var, reduced_fragment[i])

    return reduce_fused_kernel


def reduce_fused(
    x: Union[torch.Tensor, QuantTensor],
    topk_weights: Optional[torch.Tensor],
    token_topk_to_pos: torch.Tensor,
    fp8_format: str = '',
    sf: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reduce expanded expert outputs back to token-level by weighted summation.

    Args:
        x: Expanded tensor of shape (num_expanded_tokens, hidden), or a
            ``QuantTensor`` ``(data, sf)`` to include per-token sf factors.
        topk_weights: Optional top-k routing weights of shape (num_tokens, num_topk).
        token_topk_to_pos: Mapping from (token, topk) to expanded position,
            shape (num_tokens, num_topk).
        fp8_format: Optional FP8 output format (``'e4m3'`` or empty string).
        sf: Optional FP32 scalar tensor for output scaling.
        out: Optional pre-allocated output tensor of shape (num_tokens, hidden).

    Returns:
        Reduced tensor of shape (num_tokens, hidden).
    """
    if isinstance(x, tuple):
        x, x_sf = x
    else:
        x_sf = None
    num_expanded_tokens, hidden = x.shape
    num_tokens, num_topk = token_topk_to_pos.shape
    assert hidden % 256 == 0

    in_dtype = x.dtype
    out_dtype = in_dtype
    if fp8_format != '':
        if fp8_format == 'e4m3':
            out_dtype = torch.float8_e4m3fn
        else:
            assert False
    else:
        assert sf is None, 'Only FP8 output supports sf.'

    # Allocate (when needed) and launch on the device of the input tensor, which is
    # not necessarily the current CUDA device.
    device = x.device
    assert token_topk_to_pos.device == device, f'token_topk_to_pos is on {token_topk_to_pos.device}, expected {device}'
    if topk_weights is not None:
        assert topk_weights.device == device, f'topk_weights is on {topk_weights.device}, expected {device}'
    if x_sf is not None:
        assert x_sf.device == device, f'x_sf is on {x_sf.device}, expected {device}'
    if sf is not None:
        assert sf.device == device, f'sf is on {sf.device}, expected {device}'

    if out is not None:
        num_tokens_, hidden_ = out.shape
        assert num_tokens == num_tokens_ and hidden == hidden_
        assert out.device == device, f'out is on {out.device}, expected {device}'
    else:
        out = torch.empty((num_tokens, hidden), dtype=out_dtype, device=device)

    if x_sf is not None:
        num_expanded_tokens_ = x_sf.shape[0]
        assert num_expanded_tokens == num_expanded_tokens_

    with get_device_guard(device):
        kernel = get_reduce_fused_kernel(
            hidden,
            num_topk,
            T.dtype(in_dtype),
            T.dtype(out_dtype),
            sf is not None,
            topk_weights is not None,
            x_sf is not None,
        )
        if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
            print(kernel.get_kernel_source())

        if num_tokens > 0:
            kernel(x, topk_weights, token_topk_to_pos, out, sf, x_sf)

    return out

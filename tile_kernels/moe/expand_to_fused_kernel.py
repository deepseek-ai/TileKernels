import os
import torch
import tilelang
from tilelang import language as T
from typing import Optional

from tile_kernels.utils import align, ceil_div
from tile_kernels.quant.common import QuantTensor


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_expand_to_fused_kernel(
    hidden: int,
    num_topk: int,
    num_per_channels: Optional[int],
    use_tma_aligned_col_major_sf: Optional[bool],
    use_packed_ue8m0: Optional[bool],
    x_dtype: T.dtype,
    sf_dtype: T.dtype,
):
    num_threads = 64

    hidden_aligned = align(hidden, num_threads)
    if num_per_channels is not None:
        hidden_sf = ceil_div(hidden, num_per_channels)
        if use_packed_ue8m0:
            hidden_sf = ceil_div(hidden_sf, 4)
        hidden_sf_aligned = align(hidden_sf, num_threads)
    else:
        hidden_sf, hidden_sf_aligned = 1, 1

    # Runtime symbols
    sf_stride = T.dynamic('sf_stride')
    num_tokens = T.dynamic('num_tokens')
    num_expanded_tokens = T.dynamic('num_expanded_tokens')
    num_blocks = T.max(num_tokens, num_expanded_tokens)

    sf_shape = (hidden_sf, num_expanded_tokens) if use_tma_aligned_col_major_sf else (num_expanded_tokens, hidden_sf)

    @T.prim_func
    def expand_to_fused_kernel(
        x: T.Tensor[(num_tokens, hidden), x_dtype],
        x_sf: T.Tensor[(num_tokens, hidden_sf), sf_dtype],
        expanded_x: T.Tensor[(num_expanded_tokens, hidden), x_dtype],
        expanded_x_sf: T.StridedTensor[sf_shape, (sf_stride, 1), sf_dtype],
        token_topk_to_pos: T.Tensor[(num_tokens, num_topk), T.int32],
        pos_to_expert: T.Tensor[(num_expanded_tokens, ), T.int32]
    ):
        with T.Kernel(num_blocks, threads=num_threads) as (pid_token, ):
            pos_local = T.alloc_local((num_topk, ), T.int32)

            if pid_token < num_expanded_tokens:
                if pos_to_expert[pid_token] < 0:
                    for i in T.Parallel(hidden_aligned):
                        if i < hidden:
                            expanded_x[pid_token, i] = 0
                    if num_per_channels is not None:
                        for i in T.Parallel(hidden_sf_aligned):
                            if i < hidden_sf:
                                if use_tma_aligned_col_major_sf:
                                    expanded_x_sf[i, pid_token] = 0
                                else:
                                    expanded_x_sf[pid_token, i] = 0

            if pid_token >= num_tokens:
                T.thread_return()
            T.assume(pid_token < num_tokens)

            x_fragment = T.alloc_fragment((hidden_aligned, ), x_dtype)
            x_sf_fragment = T.alloc_fragment((hidden_sf_aligned, ), sf_dtype)

            T.copy(token_topk_to_pos[pid_token, 0], pos_local)
            T.copy(x[pid_token, :], x_fragment[0:hidden])
            if num_per_channels is not None:
                T.copy(x_sf[pid_token, :], x_sf_fragment[0:hidden_sf])

            for k in T.serial(num_topk):
                T.assume(pos_local[k] < num_expanded_tokens)
                if pos_local[k] >= 0:
                    for i in T.Parallel(hidden_aligned):
                        if i < hidden:
                            expanded_x[pos_local[k], i] = x_fragment[i]
                    if num_per_channels is not None:
                        for i in T.Parallel(hidden_sf_aligned):
                            if i < hidden_sf:
                                if use_tma_aligned_col_major_sf:
                                    expanded_x_sf[i, pos_local[k]] = x_sf_fragment[i]
                                else:
                                    expanded_x_sf[pos_local[k], i] = x_sf_fragment[i]

    return expand_to_fused_kernel


def expand_to_fused(x: torch.Tensor, token_topk_to_pos: torch.Tensor, pos_to_expert: torch.Tensor) -> torch.Tensor:
    """Expand token activations into the fused expert layout.

    Args:
        x: Input tensor of shape (num_tokens, hidden).
        token_topk_to_pos: Mapping from (token, topk) to expanded position,
            shape (num_tokens, num_topk).
        pos_to_expert: Mapping from expanded position to expert index,
            shape (num_expanded_tokens,).

    Returns:
        Expanded tensor of shape (num_expanded_tokens, hidden).
    """
    assert x.is_contiguous() and token_topk_to_pos.is_contiguous()
    assert x.dim() == 2 and token_topk_to_pos.dim() == 2

    num_tokens, hidden = x.shape
    num_tokens_, num_topk = token_topk_to_pos.shape
    num_expanded_tokens = pos_to_expert.shape[0]
    assert num_tokens == num_tokens_

    kernel = get_expand_to_fused_kernel(
        hidden,
        num_topk,
        None, None, None,
        T.dtype(x.dtype),
        T.dtype(x.dtype),
    )

    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    out = torch.empty((num_expanded_tokens, hidden), dtype=x.dtype, device='cuda')
    if num_tokens > 0:
        kernel(x, None, out, None, token_topk_to_pos, pos_to_expert)

    return out


def expand_to_fused_with_sf(
    x: QuantTensor,
    num_per_channels: int,
    token_topk_to_pos: torch.Tensor,
    pos_to_expert: torch.Tensor,
    use_tma_aligned_col_major_sf: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand token activations and their sf factors into the fused expert layout.

    Args:
        x: Input ``QuantTensor`` ``(data, sf)`` where data has shape
            (num_tokens, hidden) and sf has shape (num_tokens, hidden_sf).
        num_per_channels: Number of channels in each scaling block (e.g. 128).
        token_topk_to_pos: Mapping from (token, topk) to expanded position,
            shape (num_tokens, num_topk).
        pos_to_expert: Mapping from expanded position to expert index,
            shape (num_expanded_tokens,).
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major
            layout for sf factors.

    Returns:
        A tuple ``(out, out_sf)`` with expanded activation and sf-factor tensors.
    """
    x, x_sf = x
    assert x.is_contiguous() and x_sf.is_contiguous() and token_topk_to_pos.is_contiguous() and pos_to_expert.is_contiguous()
    assert x.dim() == 2 and token_topk_to_pos.dim() == 2 and pos_to_expert.dim() == 1
    assert x_sf.dtype in (torch.float32, torch.int32)
    assert num_per_channels in [32, 128]

    num_tokens, hidden = x.shape
    num_topk = token_topk_to_pos.shape[1]
    num_expanded_tokens = pos_to_expert.shape[0]
    assert num_tokens == token_topk_to_pos.shape[0]
    assert num_tokens == x_sf.shape[0]

    num_expanded_sf_tokens = align(num_expanded_tokens, 4) if use_tma_aligned_col_major_sf else num_expanded_tokens
    hidden_sf = ceil_div(hidden, num_per_channels)

    use_packed_ue8m0 = False
    if x_sf.dtype == torch.int32:
        use_packed_ue8m0 = True
        hidden_sf = ceil_div(hidden_sf, 4)
        assert use_tma_aligned_col_major_sf

    assert hidden_sf == x_sf.shape[1]

    kernel = get_expand_to_fused_kernel(
        hidden,
        num_topk,
        num_per_channels,
        use_tma_aligned_col_major_sf,
        use_packed_ue8m0,
        T.dtype(x.dtype),
        T.dtype(x_sf.dtype),
    )

    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    out = torch.empty((num_expanded_tokens, hidden), dtype=x.dtype, device='cuda')
    out_sf = torch.empty((hidden_sf, num_expanded_sf_tokens) if use_tma_aligned_col_major_sf else (num_expanded_tokens, hidden_sf), dtype=x_sf.dtype, device='cuda')
    out_sf = out_sf[:, :num_expanded_tokens] if use_tma_aligned_col_major_sf else out_sf

    if num_tokens > 0:
        kernel(x, x_sf, out, out_sf, token_topk_to_pos, pos_to_expert)
    out_sf = out_sf.T if use_tma_aligned_col_major_sf else out_sf

    return out, out_sf

import math

import torch
from tilelang import language as T

from tile_kernels.config import get_max_ub_per_vector_core, get_num_vec_cores, get_pdl, is_ascend
from tile_kernels.mhc.multilayer_recompute_cuda import mhc_multilayer_recompute_cuda
from tile_kernels.utils import align


# Mixes (pre/post/comb) stay fp32 in the Ascend kernel; reserve room for them plus
# buffer-alignment slack when sizing the activation tiles.
_ASC_MIX_SLACK_BYTES = 8 * 1024


def _asc_h_blk(hidden: int, mhc_mult: int, x_dtype: torch.dtype) -> int:
    """Hidden tile for the Ascend multilayer recompute kernel.

    bf16 keeps the historical single-tile choice. fp32 doubles the activation
    footprint, so cap h_blk by the UB budget and split hidden into tiles when
    needed (the kernel already loops over ``num_hidden_tiles``).
    """
    from tile_kernels.mhc.common_asc import _H_BLK_ALIGN

    if x_dtype == torch.bfloat16:
        return min(align(hidden, _H_BLK_ALIGN), 8192)
    assert x_dtype == torch.float32, f'unsupported dtype {x_dtype}'
    mhc_pad = align(mhc_mult, 8)
    mix_bytes = (2 * mhc_pad + 2 * mhc_pad + 2 * mhc_mult * mhc_mult) * T.float32.bytes
    budget = get_max_ub_per_vector_core(use_simt=False) - mix_bytes - _ASC_MIX_SLACK_BYTES
    bytes_per_hidden = (2 * mhc_mult + 4) * x_dtype.itemsize
    h_blk = (budget // bytes_per_hidden) // _H_BLK_ALIGN * _H_BLK_ALIGN
    assert h_blk >= _H_BLK_ALIGN, f'hidden tile does not fit UB (h_blk={h_blk}, hidden={hidden})'
    if hidden <= h_blk:
        return align(hidden, _H_BLK_ALIGN)
    for tile in range(h_blk, _H_BLK_ALIGN - 1, -_H_BLK_ALIGN):
        if hidden % tile == 0:
            return tile
    return h_blk


def _make_ptr_tables_batched(
    tensor_lists: list[list[torch.Tensor]],
    device: torch.device,
) -> list[torch.Tensor]:
    sizes = [len(tl) for tl in tensor_lists]
    total = sum(sizes)
    if total == 0:
        return [torch.empty(0, device=device, dtype=torch.int64) for _ in tensor_lists]

    pinned = torch.empty(total, dtype=torch.int64, device='cpu', pin_memory=True)
    offset = 0
    for tl in tensor_lists:
        for i, t in enumerate(tl):
            pinned[offset + i] = t.data_ptr()
        offset += len(tl)

    gpu_buf = pinned.to(device=device, non_blocking=True)

    tables: list[torch.Tensor] = []
    offset = 0
    for sz in sizes:
        tables.append(gpu_buf[offset : offset + sz])
        offset += sz
    return tables


def mhc_multilayer_recompute(
    initial_residual: torch.Tensor,
    pre_mix_list: list[torch.Tensor],
    layer_output_list: list[torch.Tensor],
    post_mix_list: list[torch.Tensor],
    comb_mix_list: list[torch.Tensor],
    layer_input_list: list[torch.Tensor],
    residual_list: list[torch.Tensor],
) -> None:
    num_layers = len(pre_mix_list)
    num_post = len(layer_output_list)
    assert num_layers == len(layer_input_list)
    assert num_post == len(post_mix_list) == len(comb_mix_list) == len(residual_list)
    assert num_post == num_layers - 1 or num_post == num_layers, (
        f'post count ({num_post}) must be num_layers-1 or num_layers (num_layers={num_layers})'
    )
    assert num_layers > 0

    mhc_mult = initial_residual.shape[-2]
    hidden = initial_residual.shape[-1]

    (
        pre_mix_ptrs,
        layer_input_ptrs,
        residual_ptrs,
        layer_output_ptrs,
        post_mix_ptrs,
        comb_mix_ptrs,
    ) = _make_ptr_tables_batched(
        [pre_mix_list, layer_input_list, residual_list, layer_output_list, post_mix_list, comb_mix_list],
        device=initial_residual.device,
    )

    if is_ascend():
        from tile_kernels.mhc.multilayer_recompute_asc import mhc_multilayer_recompute_asc

        h_blk = _asc_h_blk(hidden, mhc_mult, initial_residual.dtype)
        mhc_multilayer_recompute_asc(
            initial_residual.view(-1, mhc_mult, hidden),
            pre_mix_ptrs,
            layer_output_ptrs,
            post_mix_ptrs,
            comb_mix_ptrs,
            layer_input_ptrs,
            residual_ptrs,
            h_blk=h_blk,
            num_vec_cores=get_num_vec_cores(),
            x_dtype=T.dtype(initial_residual.dtype),
        )
    else:
        h_blk = math.gcd(512, hidden)
        mhc_multilayer_recompute_cuda(
            initial_residual.view(-1, mhc_mult, hidden),
            pre_mix_ptrs,
            layer_output_ptrs,
            post_mix_ptrs,
            comb_mix_ptrs,
            layer_input_ptrs,
            residual_ptrs,
            n_thr=64,
            h_blk=h_blk,
            x_dtype=T.dtype(initial_residual.dtype),
            use_pdl=get_pdl(),
        )

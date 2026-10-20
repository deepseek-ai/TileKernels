from typing import Optional, Sequence

import torch

from tile_kernels.config import get_num_vec_cores, get_pdl, is_ascend
from tile_kernels.quant.batched_transpose_weight_sf_cuda import get_batched_transpose_weight_sf_kernel_cuda
from tile_kernels.quant.common import get_packed_ue8m0_pack_factor, get_packed_ue8m0_torch_dtype
from tile_kernels.utils import ceil_div


def batched_transpose_weight_sf(
    sf: Sequence[torch.Tensor],
    block_size: int,
    transpose_sf1s: Sequence[bool],
    outs: Sequence[Optional[tuple[torch.Tensor, torch.Tensor]]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Transform per-weight fp32 power-of-two scales into deep_gemm UE8M0 sf0_out/sf1_out.

    Args:
        sf: Per-weight fp32 scales with shape ``(n / block_size, k / block_size)`` or
            ``(num_groups, n / block_size, k / block_size)``; ``n``/``k`` are derived from the
            shape and must be multiples of 128.
        block_size: Scaling block size of the weight (32 or 128).
        transpose_sf1s: Per-weight flag; when set, ``sf1_out`` is returned transposed as
            ``(num_packed_sf1, k)`` instead of ``(k, num_packed_sf1)``.
        outs: Per-weight ``(sf0_out, sf1_out)`` from a previous call; non-``None``
            entries are rewritten in place instead of allocating new tensors.

    Returns:
        One ``(sf0_out, sf1_out)`` pair per weight.
    """
    assert block_size in (32, 128)
    assert not is_ascend() or block_size == 32, 'Ascend only supports block_size=32'
    assert len(sf) == len(transpose_sf1s) == len(outs)
    if len(sf) == 0:
        return []

    device = sf[0].device
    # MUSA reuses the CUDA code path: the kernel body only moves fp32 exponent
    # bytes through shared memory (no CUDA-only intrinsic), so the device guard
    # is a backends-with-a-TileLang-target check rather than a CUDA check.
    assert device.type in ('cuda', 'npu', 'musa')
    pack_factor = get_packed_ue8m0_pack_factor()
    sf_out_dtype = get_packed_ue8m0_torch_dtype()
    num_sf_per_block = 32 if is_ascend() else 16
    sf_out_pool_size = num_tasks = 0
    weight_infos, num_blocks, pool_offsets = [], [], []

    for weight_sf, out in zip(sf, outs):
        assert weight_sf.dtype == torch.float32 and weight_sf.is_contiguous() and weight_sf.device == device
        assert weight_sf.dim() in (2, 3)
        num_groups = weight_sf.shape[0] if weight_sf.dim() == 3 else 1
        num_sf_n, num_sf_k = weight_sf.shape[-2:]
        n, k = num_sf_n * block_size, num_sf_k * block_size
        assert num_groups > 0 and n % 128 == 0 and k % 128 == 0

        weight_infos.append((n, k, num_groups))
        num_blocks.append(num_tasks)
        num_tasks += num_groups * ceil_div(num_sf_n, num_sf_per_block) * ceil_div(num_sf_k, num_sf_per_block)
        if out is None:
            num_packed_sf0, num_packed_sf1 = ceil_div(num_sf_k, pack_factor), ceil_div(num_sf_n, pack_factor)
            sf0_out_offset = sf_out_pool_size
            sf1_out_offset = sf0_out_offset + num_groups * num_packed_sf0 * n
            sf_out_pool_size = sf1_out_offset + num_groups * num_packed_sf1 * k
            pool_offsets.append((sf0_out_offset, sf1_out_offset))
        else:
            pool_offsets.append(None)

    sf_out_pool = torch.empty(sf_out_pool_size, dtype=sf_out_dtype, device=device)
    ptrs, sf_pairs = [], []
    for weight_sf, transpose_sf1, out, out_offsets in zip(sf, transpose_sf1s, outs, pool_offsets):
        group_shape = weight_sf.shape[:-2]
        num_sf_n, num_sf_k = weight_sf.shape[-2:]
        n, k = num_sf_n * block_size, num_sf_k * block_size
        num_packed_sf0, num_packed_sf1 = ceil_div(num_sf_k, pack_factor), ceil_div(num_sf_n, pack_factor)
        sf1_shape, sf1_stride = ((num_packed_sf1, k), (k, 1)) if transpose_sf1 else ((k, num_packed_sf1), (1, k))
        out_layouts = (
            ((*group_shape, n, num_packed_sf0), ((num_packed_sf0 * n,) if group_shape else ()) + (1, n)),
            ((*group_shape, *sf1_shape), ((num_packed_sf1 * k,) if group_shape else ()) + sf1_stride),
        )
        if out is None:
            out = tuple(sf_out_pool.as_strided(shape, strides, offset) for (shape, strides), offset in zip(out_layouts, out_offsets))
        else:
            for sf_out, (shape, strides) in zip(out, out_layouts):
                assert sf_out.dtype == sf_out_dtype and sf_out.device == device and sf_out.data_ptr() % 32 == 0
                assert sf_out.shape == shape and sf_out.stride() == strides, f'{sf_out.shape=} {sf_out.stride()=} expected {shape} {strides}'
        ptrs.append((weight_sf.data_ptr(), out[0].data_ptr(), out[1].data_ptr()))
        sf_pairs.append(out)

    num_blocks = torch.tensor(num_blocks, dtype=torch.int32, device=device)
    weight_infos = torch.tensor(weight_infos, dtype=torch.int32, device=device)
    ptrs = torch.tensor(ptrs, dtype=torch.int64, device=device)
    if is_ascend():
        from tile_kernels.quant.batched_transpose_weight_sf_asc import get_batched_transpose_weight_sf_kernel_asc

        kernel = get_batched_transpose_weight_sf_kernel_asc(block_size, get_num_vec_cores())
    else:
        kernel = get_batched_transpose_weight_sf_kernel_cuda(block_size, use_pdl=get_pdl())
    kernel(num_blocks, weight_infos, ptrs, num_tasks)
    return sf_pairs

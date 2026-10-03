from typing import Optional

import torch
from tilelang import language as T

from tile_kernels.config import get_num_vec_cores, get_pdl, is_ascend
from tile_kernels.quant.common import *
from tile_kernels.quant.norm_forward_cuda import get_norm_forward_and_per_token_cast_kernel_cuda
from tile_kernels.utils import align


def _overlaps(a: torch.Tensor, b: torch.Tensor) -> bool:
    if not a.numel() or not b.numel():
        return False
    a_start, b_start = a.data_ptr(), b.data_ptr()
    return a_start < b_start + b.numel() * b.element_size() and b_start < a_start + a.numel() * a.element_size()


def norm_forward_and_per_token_cast_impl(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    fmt: str,
    num_per_channels: Optional[int],
    residual: Optional[torch.Tensor] = None,
    out_scale: float = 1.0,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    with_out_bf16: bool = False,
    out: Optional[torch.Tensor] = None,
    out_sf: Optional[torch.Tensor] = None,
    out_bf16: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    mega_moe_sf: Optional[torch.Tensor] = None,
    mega_moe_block_m: int = 0,
) -> tuple[Union[torch.Tensor, QuantTensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    assert x.ndim in (2, 3) and x.dtype in (torch.bfloat16, torch.float32) and x.stride(-1) == 1
    num_tokens, hidden = x.shape[-2:]
    batch = x.shape[0] if x.ndim == 3 else 1
    assert hidden > 0
    if is_ascend():
        # BF16 padding uses signed 16-bit lane indices in the Ascend kernel.
        assert x.dtype != torch.bfloat16 or align(hidden, 128) <= 32768, 'Ascend BF16 norm hidden size exceeds 16-bit lane indexing'
    else:
        assert hidden % 16 == 0 and hidden <= 8192

    def batched(tensor):
        return tensor.unsqueeze(0) if tensor is not None and x.ndim == 2 else tensor

    if weight is not None:
        assert weight.shape == (hidden,) and weight.dtype == x.dtype and weight.device == x.device and weight.is_contiguous()

    valid_quant_groups = (32,) if is_ascend() else (32, 128)
    assert (fmt == 'e4m3' and num_per_channels in valid_quant_groups) or (fmt, num_per_channels) in [('bf16', None), ('fp32', None)]
    out_config = get_cast_output_config(fmt, (1, num_per_channels or 1), use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0)
    assert out_config.with_sf or out_config.torch_dtype == x.dtype, 'the unquantized output must match the input dtype'
    assert not out_config.with_sf or hidden % num_per_channels == 0
    assert not with_out_bf16 or out_config.with_sf, 'the extra output is only needed with casting'
    assert (mega_moe_sf is None) == (mega_moe_block_m == 0)
    if mega_moe_sf is not None:
        # INT32 (num_tokens, hidden // 128) view of M-major rows, each row padded per `mega_moe_block_m` tokens
        assert x.ndim == 2 and mega_moe_block_m > 0
        assert out_config.with_sf and num_per_channels == 32 and round_sf and hidden % 128 == 0
        assert mega_moe_sf.device == x.device
        assert mega_moe_sf.dtype == torch.int32 and mega_moe_sf.shape == (num_tokens, hidden // 128) and mega_moe_sf.stride(0) == 1
        num_mega_moe_sf_rows = mega_moe_sf.stride(1)
        assert ceil_div(num_tokens, mega_moe_block_m) * align(mega_moe_block_m, 128) <= num_mega_moe_sf_rows

    # Allocate outputs
    if out is None:
        out = torch.empty(x.shape, dtype=out_config.torch_dtype, device=x.device)
    if out_config.with_sf:
        if out_sf is None:
            alloc_shape = (num_tokens, hidden)
            if x.ndim == 3:
                if use_tma_aligned_col_major_sf:
                    # Keep packed hidden padding and TMA token alignment separate for each batch.
                    sf_pack = get_packed_ue8m0_pack_factor() if use_packed_ue8m0 else 1
                    alloc_shape = (num_tokens, batch * align(hidden, num_per_channels * sf_pack))
                else:
                    alloc_shape = (batch * num_tokens, hidden)
            out_sf = alloc_scaling_factors(alloc_shape, out_config, x.device)
            if x.ndim == 3:
                sf_rows, sf_cols = get_sf_shape((num_tokens, hidden), out_config)
                # Explicit strides also preserve the zero-token layout required by dtype views.
                out_sf = out_sf.as_strided((batch, sf_rows, sf_cols), (sf_rows * out_sf.stride(0), out_sf.stride(0), 1))
        else:
            # External buffers only support row-major SF
            assert not use_tma_aligned_col_major_sf
            if use_packed_ue8m0:
                assert out_sf.dtype == get_packed_ue8m0_torch_dtype()
                out_sf = out_sf.view(torch.uint8)
            expected_sf_shape = get_sf_shape((num_tokens, hidden), out_config)
            assert out_sf.shape == (*x.shape[:-2], *expected_sf_shape) and out_sf.stride(-1) == 1
            assert out_sf.dtype == out_config.sf_torch_dtype and out_sf.device == x.device
    else:
        assert out_sf is None
    if with_out_bf16:
        if out_bf16 is None:
            out_bf16 = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    else:
        assert out_bf16 is None
    residual_out = torch.empty_like(x) if residual is not None and residual_out is None else residual_out
    if out_bf16 is not None:
        assert out_bf16.shape == x.shape and out_bf16.dtype == torch.bfloat16 and out_bf16.device == x.device and out_bf16.stride(-1) == 1
    for tensor in (residual, residual_out):
        if tensor is not None:
            assert tensor.shape == x.shape and tensor.dtype == x.dtype and tensor.device == x.device and tensor.stride(-1) == 1
    assert out.shape == x.shape and out.dtype == out_config.torch_dtype and out.device == x.device and out.stride(-1) == 1
    rstd = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)

    # Get kernel implement
    kernel_kwargs = dict(
        hidden=hidden,
        with_weight=weight is not None,
        with_residual=residual is not None,
        eps=eps,
        out_scale=out_scale,
        x_dtype=T.dtype(x.dtype),
        out_config=out_config,
        with_out_bf16=with_out_bf16,
        mega_moe_block_m=mega_moe_block_m,
    )
    if batch * num_tokens > 0:
        if is_ascend():
            from tile_kernels.quant.norm_forward_asc import get_norm_forward_kernel_asc

            kernel = get_norm_forward_kernel_asc(num_tokens=num_tokens, num_vec_cores=get_num_vec_cores(), **kernel_kwargs)
        else:
            for tensor in (x, out, out_bf16, residual, residual_out):
                if tensor is not None:
                    assert all(stride % 16 == 0 for stride in tensor.stride()[:-1])
            kernel = get_norm_forward_and_per_token_cast_kernel_cuda(**kernel_kwargs, use_pdl=get_pdl())
        if mega_moe_sf is not None:
            mega_moe_sf = mega_moe_sf.mT.view(torch.uint8).as_strided((num_mega_moe_sf_rows, hidden // 128, 4), (4, num_mega_moe_sf_rows * 4, 1))
        args = (
            batched(x),
            batched(out),
            batched(out_sf),
            batched(out_bf16),
            mega_moe_sf,
            weight,
            batched(rstd),
            batched(residual),
            batched(residual_out),
        )

        kernel(*args)

    norm_input = x if residual is None else residual_out
    if out_config.with_sf:
        if x.ndim == 3 and use_tma_aligned_col_major_sf:
            # Internally allocated column-major SF can share a 2D view for the common epilogue.
            sf_rows, sf_cols = out_sf.shape[-2:]
            out_sf = out_sf.as_strided((batch * sf_rows, sf_cols), (out_sf.stride(1), 1))
            out_sf = cast_epilogue(out_sf, out_config)
            out_sf = out_sf.as_strided((batch, out_sf.shape[0], sf_rows), (sf_rows * out_sf.stride(1), *out_sf.stride()))
        else:
            # Row-major packing preserves batch strides, including gaps in external buffers.
            out_sf = cast_epilogue(out_sf, out_config)
        return (out, out_sf), out_bf16, rstd, norm_input
    return out, None, rstd, norm_input


def norm_forward_and_per_token_cast(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    fmt: str,
    num_per_channels: int,
    residual: Optional[torch.Tensor] = None,
    out_scale: float = 1.0,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    with_out_bf16: bool = False,
    out: Optional[QuantTensor] = None,
    out_bf16: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    mega_moe_sf: Optional[torch.Tensor] = None,
    mega_moe_block_m: int = 0,
) -> tuple[QuantTensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """RMSNorm with optional input-dtype residual addition, cast to FP8 with per-token (row-wise) scaling factors.

    The normalized output is rounded to x.dtype first, so the result equals `per_token_cast` of `norm_forward`.

    Args:
        x: (num_tokens, hidden) or (batch, num_tokens, hidden), BF16/FP32; last dimension contiguous.
        weight: Optional (hidden,) tensor matching x.dtype/device; None means unit weight.
        eps: Epsilon added to the mean square before reciprocal square root.
        fmt: Target quantized format (``'e4m3'``).
        num_per_channels: Must be 32 on Ascend and 32 or 128 on CUDA; hidden must be divisible by it.
        residual: Optional tensor matching x; residual_out always rounds to x.dtype.
        out_scale: FP32 multiplier applied to weight before the output product.
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major sf factors.
        round_sf: Whether to round scaling factors to powers of two.
        use_packed_ue8m0: Pack SF bytes into int16 on Ascend and int32 on CUDA.
        with_out_bf16: Whether to also return the normalized output as BF16.
        out: Optional pre-allocated QuantTensor (data, sf); the sf buffer must be row-major.
        out_bf16: Optional BF16 buffer for the normalized output, requires `with_out_bf16`.
        residual_out: Optional buffer for x + residual; ignored without residual.
        mega_moe_sf: Optional INT32 (num_tokens, hidden // 128) M-major view of the mega MoE shared activation sf buffer,
            receiving the packed UE8M0 sf of 32 channels as well: tokens are split into `mega_moe_block_m` chunks, each padded
            to a multiple of 128 rows, and the rows of every 128 are transposed as 4 x 32. Rows of padding are untouched.
        mega_moe_block_m: Token block size of the mega MoE kernel, required with `mega_moe_sf`.

    Returns:
        A tuple ``((out, out_sf), out_bf16, rstd, norm_input)``: the quantized output and sf factors, the normalized
        BF16 output (None without `with_out_bf16`), FP32 rstd of shape x.shape[:-1], and the residual sum in x.dtype
        (aliasing x without residual).
    """
    out, out_sf = (None, None) if out is None else out
    return norm_forward_and_per_token_cast_impl(
        x,
        weight,
        eps,
        fmt,
        num_per_channels,
        residual,
        out_scale,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
        with_out_bf16=with_out_bf16,
        out=out,
        out_sf=out_sf,
        out_bf16=out_bf16,
        residual_out=residual_out,
        mega_moe_sf=mega_moe_sf,
        mega_moe_block_m=mega_moe_block_m,
    )


def add_rmsnorm_forward_and_per_token_cast(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    fmt: str,
    num_per_channels: int,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> QuantTensor:
    """Add ``x`` to ``residual`` in place, RMS-normalize, and cast to FP8."""
    assert x.ndim == 2 and x.is_contiguous() and x.dtype in (torch.bfloat16, torch.float32)
    assert residual.shape == x.shape and residual.dtype == x.dtype and residual.device == x.device and residual.is_contiguous()
    assert x.device.type == 'cuda' and not _overlaps(x, residual)
    assert weight.shape == (x.shape[1],) and weight.dtype == x.dtype and weight.device == x.device and weight.is_contiguous()
    assert not _overlaps(residual, weight)

    quant_out, _, _, norm_input = norm_forward_and_per_token_cast(
        x,
        weight,
        eps,
        fmt,
        num_per_channels,
        residual=residual,
        residual_out=residual,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )
    assert norm_input is residual
    return quant_out


def norm_forward(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    residual: Optional[torch.Tensor] = None,
    out_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RMSNorm with optional input-dtype residual addition.

    Args:
        x: (num_tokens, hidden) or (batch, num_tokens, hidden), BF16/FP32; last dimension contiguous.
        weight: Optional (hidden,) tensor matching x.dtype/device; None means unit weight.
        eps: Epsilon added to the mean square before reciprocal square root.
        residual: Optional tensor matching x; residual_out always rounds to x.dtype.
        out_scale: FP32 multiplier applied to weight before the output product.
        out: Optional output buffer matching x, with non-overlapping rows.
        residual_out: Optional buffer for x + residual; ignored without residual.

    Supports positive hidden sizes and empty batches. CUDA requires hidden divisible by 16,
    limits hidden to 8192, and requires row/batch strides divisible by 16.
    Ascend requires the padded row and enabled fusion buffers to fit in UB;
    BF16 also limits the padded hidden size to 32768 for lane indexing.
    Output buffers must not overlap inputs or each other.

    Returns:
        Output, FP32 rstd of shape x.shape[:-1], and the residual sum in x.dtype.
        Without residual the third return value aliases x.
    """
    assert x.dtype in (torch.bfloat16, torch.float32)
    out, _, rstd, norm_input = norm_forward_and_per_token_cast_impl(
        x,
        weight,
        eps,
        'bf16' if x.dtype == torch.bfloat16 else 'fp32',
        None,
        residual,
        out_scale,
        out=out,
        residual_out=residual_out,
    )
    return out, rstd, norm_input

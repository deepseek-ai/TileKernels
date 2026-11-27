import torch
from tilelang import language as T

from tile_kernels.config import get_num_vec_cores, get_pdl
from tile_kernels.quant.common import get_cast_output_config
from tile_kernels.quant.per_token_cast_and_cast_back_cuda import get_per_token_cast_and_cast_back_kernel_cuda


def per_token_cast_and_cast_back(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    round_sf: bool = False,
    use_e4m3_sf: bool = False,
):
    """Apply an in-place lossy per-token cast-and-cast-back to ``x``.

    Args:
        x: Input 2D tensor of shape (num_tokens, hidden). The innermost stride must
            be 1; ``x.stride(0)`` may exceed ``hidden`` (non-contiguous rows allowed).
        fmt: Quantized format used for cast-and-cast-back (``'e4m3'`` or ``'e2m1'``).
        num_per_channels: Number of channels in each scaling block (16 or 32).
            On Ascend, only 16 and 32 are supported; 16 requires use_e4m3_sf=True.
        round_sf: Whether to round scaling factors to powers of two.
        use_e4m3_sf: Whether to round the internal scale factors to E4M3.
            On Ascend this requires num_per_channels=16 and hidden divisible by 128.
    """
    assert x.dim() == 2
    assert x.stride(1) == 1
    assert fmt in ('e4m3', 'e2m1')
    assert num_per_channels in (16, 32), 'num_per_channels must be 16 or 32'
    assert not (round_sf and use_e4m3_sf)

    num_tokens, hidden = x.shape
    assert hidden % num_per_channels == 0
    assert hidden % 64 == 0

    out_config = get_cast_output_config(fmt, (1, num_per_channels), round_sf=round_sf, use_e4m3_sf=use_e4m3_sf)
    assert out_config.sf_block[0] == 1

    if x.device.type == 'npu':
        from tile_kernels.quant.per_token_cast_and_cast_back_asc import get_per_token_cast_and_cast_back_kernel_asc

        kernel = get_per_token_cast_and_cast_back_kernel_asc(
            hidden=hidden,
            token_stride=x.stride(0),
            dtype=T.dtype(x.dtype),
            out_config=out_config,
            num_vec_cores=get_num_vec_cores(),
        )
    else:
        kernel = get_per_token_cast_and_cast_back_kernel_cuda(
            hidden=hidden,
            token_stride=x.stride(0),
            dtype=T.dtype(x.dtype),
            out_config=out_config,
            use_pdl=get_pdl(),
        )

    if num_tokens > 0:
        kernel(x)

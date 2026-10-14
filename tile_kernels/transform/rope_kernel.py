import torch
from tilelang import language as T

from tile_kernels.config import get_pdl, is_ascend

from tile_kernels.transform.rope_cuda import get_rope_kernel_cuda


def apply_rotary(
    query: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    key: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    interleaved: bool = False,
    conjugate: bool = False,
    seqlen_offset: int = 0,
) -> None:
    """Apply rotary embedding to query and optional key in place.

    Args:
        query: ``[num_tokens, num_heads, rot_dim]`` or
            ``[batch, seqlen, num_heads, rot_dim]``. CUDA supports
            fp16/bf16/fp32; Ascend supports bf16/fp32.
        cos_sin_cache: ``[seqlen_ro, rot_dim]``, contiguous in the last
            dimension and fp32 — cos in ``[:, :rot_dim // 2]``, sin in
            ``[:, rot_dim // 2:]``.
        key: like ``query`` with ``num_kv_heads``; same dtype (or ``None``).
        positions: optional int32/int64 row indices into ``cos_sin_cache``;
            ``[num_tokens]`` for 3D inputs or ``[batch, seqlen]`` for 4D
            inputs. ``seqlen_offset`` is added to every index; without
            ``positions``, row ``seqlen_offset + token`` is used.
        interleaved: GPT-J interleaved pairing when True, else NeoX half-split.
        conjugate: apply the inverse rotation.
        seqlen_offset: scalar offset added to positions.

    CUDA supports ``rot_dim`` values 32, 64, 128, and 256; Ascend supports
    64 and 128. The tensor's last dimension must equal ``rot_dim``. Express
    partial RoPE with a view such as ``query[..., :rot_dim]`` or
    ``query[..., -rot_dim:]``.
    """
    assert query.ndim in (3, 4)
    assert isinstance(seqlen_offset, int)
    if key is not None:
        assert key.ndim == query.ndim
    assert cos_sin_cache.ndim == 2

    if query.ndim == 3:
        seqlen, num_heads, _ = query.shape
        if positions is not None:
            assert positions.shape == (seqlen,)
            positions = positions.unsqueeze(0)
        query = query.unsqueeze(0)
        if key is not None:
            num_kv_heads = key.shape[1]
            key = key.unsqueeze(0)
    else:
        batch, seqlen, num_heads, _ = query.shape
        if positions is not None:
            assert positions.shape == (batch, seqlen)
        if key is not None:
            num_kv_heads = key.shape[2]

    _, rot_dim = cos_sin_cache.shape

    ascend = is_ascend()
    assert rot_dim in ((64, 128) if ascend else (32, 64, 128, 256))
    assert query.shape[-1] == rot_dim
    assert query.stride(-1) == 1
    assert cos_sin_cache.dtype == torch.float32
    assert cos_sin_cache.stride(1) == 1
    assert query.dtype in ((torch.bfloat16, torch.float32) if ascend else (torch.float16, torch.bfloat16, torch.float32))
    assert positions is None or positions.dtype in (torch.int32, torch.int64)
    assert positions is None or positions.stride(-1) == 1
    if key is not None:
        assert key.shape[-1] == rot_dim
        assert key.dtype == query.dtype
        assert key.stride(-1) == 1

    launch_specs = [(query, num_heads)]
    if key is not None:
        launch_specs.append((key, num_kv_heads))

    for x, nheads in launch_specs:
        if x.numel() == 0:
            continue
        if ascend:
            from tile_kernels.transform.rope_asc import get_rope_kernel_asc

            kernel = get_rope_kernel_asc(
                T.dtype(x.dtype),
                T.int32 if positions is None else T.dtype(positions.dtype),
                nheads,
                rot_dim,
                positions is not None,
                interleaved,
                conjugate,
                cos_sin_cache.stride(0),
                x.stride(1),
                x.stride(2),
            )
        else:
            batch_is_one = x.shape[0] == 1
            assert batch_is_one or x.stride(0) % 8 == 0
            kernel = get_rope_kernel_cuda(
                T.dtype(x.dtype),
                T.int32 if positions is None else T.dtype(positions.dtype),
                nheads,
                rot_dim,
                positions is not None,
                batch_is_one,
                interleaved,
                conjugate,
                cos_sin_cache.stride(0),
                x.stride(1),
                x.stride(2),
                use_pdl=get_pdl(),
            )
        kernel(x, cos_sin_cache, positions, seqlen_offset)

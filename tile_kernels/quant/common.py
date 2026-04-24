from dataclasses import dataclass, replace
from typing import Optional, Union

import torch
from tilelang import language as T
from tilelang.utils.target import determine_target

from tile_kernels.quant.types import QuantTensor
from tile_kernels.utils import align, ceil_div


def get_best_vectorize_size(dtype: T.dtype) -> int:
    target = determine_target(return_object=True)
    if target.kind.name == 'hip':
        # AMD GPU: use 16-byte vectorized loads (128-bit, supported on gfx90x/gfx950)
        return 16 // dtype.bytes
    from tilelang.contrib import nvcc
    ver = nvcc.get_target_compute_version(target)  # e.g. "8.6"
    major, _ = nvcc.parse_compute_version(ver)
    return (16 if major < 10 else 32) // dtype.bytes


@dataclass(frozen=True)
class BaseCastConfig:
    torch_dtype: torch.dtype = torch.float8_e4m3fn
    sf_block: tuple[int, int] = (1, 1)
    use_tma_aligned_col_major_sf: bool = False
    use_packed_ue8m0: bool = False

    @property
    def dtype(self) -> T.dtype:
        return T.dtype(self.torch_dtype) if self.torch_dtype != torch.int8 else T.float4_e2m1fn

    @property
    def sf_torch_dtype(self) -> torch.dtype:
        return torch.uint8 if self.use_packed_ue8m0 else torch.float32

    @property
    def sf_dtype(self) -> T.dtype:
        return T.dtype(self.sf_torch_dtype)


@dataclass(frozen=True)
class CastInputConfig(BaseCastConfig):
    with_sf: bool = True


@dataclass(frozen=True)
class CastOutputConfig(BaseCastConfig):
    round_sf: bool = False
    custom_clamp_min_value: Optional[float] = None

    @property
    def clamp_min_value(self) -> float:
        if self.custom_clamp_min_value is not None:
            return self.custom_clamp_min_value
        elif self.dtype == T.float8_e4m3fn:
            return 1e-4
        elif self.dtype == T.float4_e2m1fn:
            return T.max_value(self.dtype) * (2**-126)
        else:
            raise ValueError(f'Unsupported dtype {self.dtype}')


def get_cast_input_and_config(
    x: Union[torch.Tensor, QuantTensor],
    sf_block: Optional[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor, CastInputConfig]:
    if isinstance(x, tuple):
        assert isinstance(sf_block, tuple)
        x, x_sf = x
        config = CastInputConfig(torch_dtype=x.dtype, with_sf=True, sf_block=sf_block)
        assert isinstance(x, torch.Tensor) and isinstance(x_sf, torch.Tensor)
        assert x.dtype in (torch.float8_e4m3fn, torch.int8, torch.uint8)

        if x_sf.stride(0) == 1:
            config = replace(config, use_tma_aligned_col_major_sf=True)
            x_sf = x_sf.T
            if x_sf.dtype == torch.int32:
                config = replace(config, use_packed_ue8m0=True)
                x_sf = x_sf.view(torch.uint8)
        else:
            assert x_sf.stride(1) == 1
            assert x_sf.dtype == torch.float32
        return x, x_sf, config
    else:
        config = CastInputConfig(torch_dtype=x.dtype, with_sf=False)
        assert sf_block is None
        assert isinstance(x, torch.Tensor)
        assert x.dtype in (torch.bfloat16, torch.float32)
        return x, None, config


def get_cast_output_config(
    fmt: str,
    sf_block: tuple[int, int],
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    custom_clamp_min_value: Optional[float] = None,
) -> CastOutputConfig:
    assert fmt in ('e5m6', 'e4m3', 'e2m1')
    mapping = {
        'e5m6': torch.uint32,
        'e4m3': torch.float8_e4m3fn,
        'e2m1': torch.int8,
    }
    config = CastOutputConfig(
        torch_dtype=mapping[fmt],
        sf_block=sf_block,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
        custom_clamp_min_value=custom_clamp_min_value,
    )
    return config


def get_logical_hidden(hidden: int, dtype: torch.dtype) -> int:
    """
    Compute hidden size when `torch.int8` is used for packing FP4
    """
    return hidden if dtype != torch.int8 else hidden * 2


def get_physical_hidden(hidden: int, dtype: torch.dtype) -> int:
    """
    Compute hidden size when `torch.int8` is used for packing FP4
    """
    return hidden if dtype != torch.int8 else hidden // 2


def get_sf_shape(shape: tuple[int, int], config: BaseCastConfig) -> tuple[int, int]:
    num_block_m = ceil_div(shape[0], config.sf_block[0])
    num_block_k = ceil_div(shape[1], config.sf_block[1])
    # num_block_m = align(num_block_m, 4) if use_tma_aligned_col_major_sf else num_block_m
    # For UE8M0, we must use col-major SF, and 4 UE8M0 are expanded into the inner dim (token)
    if config.use_packed_ue8m0:
        num_block_m = num_block_m * 4
        num_block_k = ceil_div(num_block_k, 4)
    return (num_block_k, num_block_m) if config.use_tma_aligned_col_major_sf else (num_block_m, num_block_k)


def alloc_scaling_factors(
    shape: tuple[int, int],
    out_config: BaseCastConfig,
    device: torch.device = 'cuda',
) -> torch.Tensor:
    """
    Allocate scaling factors for quantization.
    """
    sf_shape = get_sf_shape(shape, out_config)

    # For col-major SF, TMA must be aligned into 16 bytes
    aligned_sf_shape = sf_shape[1]
    if out_config.use_tma_aligned_col_major_sf:
        aligned_sf_shape = align(sf_shape[1], 16 if out_config.use_packed_ue8m0 else 4)

    scaling_factor = torch.empty(
        size=(sf_shape[0], aligned_sf_shape),
        dtype=out_config.sf_torch_dtype,
        device=device,
    )

    if out_config.use_tma_aligned_col_major_sf:
        scaling_factor = scaling_factor[:, : sf_shape[1]]

    return scaling_factor


def cast_epilogue(
    out_sf: torch.Tensor,
    num_tokens: int,
    hidden: int,
    config: BaseCastConfig,
) -> torch.Tensor:
    """Post-process the sf-factor tensor after a cast kernel launch.

    Args:
        out_sf: Raw sf-factor tensor produced by the kernel.
        num_tokens: Number of tokens in the original input.
        hidden: Hidden dimension size of the original input.
        config: Cast configuration used during the kernel launch.

    Returns:
        Corrected sf-factor tensor with proper layout and shape.
    """
    # Make corrected SF tensor
    if config.use_packed_ue8m0:
        if num_tokens == 0:
            out_sf = torch.empty((out_sf.shape[0], out_sf.shape[1] // 4), dtype=torch.int32, device=out_sf.device)
        else:
            out_sf = out_sf.view(dtype=torch.int32)
    out_sf = out_sf.T if config.use_tma_aligned_col_major_sf else out_sf
    out_sf = out_sf[: ceil_div(num_tokens, config.sf_block[0]), :]
    return out_sf


@T.macro
def get_sf_and_inv(amax: float, out_config: CastOutputConfig):
    # Clamp with min value
    clamped_amax = T.max(amax, out_config.clamp_min_value)

    max_value = T.max_value(out_config.dtype)
    sf = T.alloc_var(T.float32)
    sf = clamped_amax / max_value
    if not out_config.round_sf:
        return sf, max_value / clamped_amax

    # Round into 2's power
    bits = T.reinterpret(sf, T.uint32)
    # amax >= 1e-4 ensures sign bit = 0 and bits != 0 (no denorm/zero).
    # `(bits - 1) >> 23 + 1` gives ceil(log2).
    exp_sf = ((bits - 1) >> 23) + 1 - 127
    sf_inv = T.reinterpret((127 - exp_sf) << 23, T.float32)
    if out_config.use_packed_ue8m0:
        return T.uint8(exp_sf + 127), sf_inv
    else:
        return T.reinterpret((127 + exp_sf) << 23, T.float32), sf_inv


@T.macro
def load_sf(tensor: T.Tensor, m_idx: int, k_idx: int, config: BaseCastConfig):
    if config.use_packed_ue8m0:
        return tensor[k_idx // 4, m_idx * 4 + k_idx % 4]
    elif config.use_tma_aligned_col_major_sf:
        return tensor[k_idx, m_idx]
    else:
        return tensor[m_idx, k_idx]


@T.macro
def transform_sf(sf: Union[T.float32, T.uint8], config: BaseCastConfig) -> T.float32:
    if config.use_packed_ue8m0:
        return T.reinterpret(T.uint32(sf) << 23, T.float32)
    else:
        return sf


@T.macro
def store_sf(tensor: T.Tensor, sf: Union[T.float32, T.uint8], m_idx: int, k_idx: int, config: BaseCastConfig):
    if config.use_packed_ue8m0:
        tensor[k_idx // 4, m_idx * 4 + k_idx % 4] = sf
    elif config.use_tma_aligned_col_major_sf:
        tensor[k_idx, m_idx] = sf
    else:
        tensor[m_idx, k_idx] = sf


def unpack_from_e2m1fn_x2(x: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Decode a uint8/int8 tensor of packed fp4 values to float32. Used mainly for debugging.

    The input tensor is expected to have shape [..., 2 * K] where the last dimension contains
    pairs of packed fp4 values.
    The output tensor will have shape [..., K] with the decoded float32 values.
    """
    assert x.dtype == torch.int8 or x.dtype == torch.uint8

    if x.ndim == 0:
        raise ValueError('x must have at least 1 dimension so the last dim can be doubled')

    lo = (x & 0x0F).to(torch.int16)
    hi = ((x >> 4) & 0x0F).to(torch.int16)

    def decode_fp4_e2m1(n: torch.Tensor) -> torch.Tensor:
        # n in [0..15], layout: s(1) | e(2) | m(1)
        s = (n >> 3) & 0x1
        e = (n >> 1) & 0x3
        m = n & 0x1

        sign = torch.where(
            s == 1,
            torch.tensor(-1.0, device=n.device),
            torch.tensor(1.0, device=n.device),
        )

        bias = 1

        # subnormal/zero: e==0 -> value = sign * 2^(1-bias) * (m/2)
        # bias=1 => 2^(0)=1 => {0, 0.5}
        sub = (m.to(torch.float32) * 0.5) * (2.0 ** (1 - bias))

        # normal: e in {1,2,3} -> value = sign * 2^(e-bias) * (1 + m/2)
        norm = (1.0 + m.to(torch.float32) * 0.5) * torch.pow(
            torch.tensor(2.0, device=n.device),
            (e - bias).to(torch.float32),
        )

        val = torch.where(e == 0, sub, norm)
        return (val * sign).to(out_dtype)

    flo = decode_fp4_e2m1(lo)
    fhi = decode_fp4_e2m1(hi)

    # (..., L) -> (..., L, 2) -> (..., 2L)
    y = torch.stack([flo, fhi], dim=-1).reshape(*x.shape[:-1], x.shape[-1] * 2)
    return y

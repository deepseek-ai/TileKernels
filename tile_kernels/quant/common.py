import functools
from dataclasses import dataclass
from typing import Optional, Union

import torch
from tilelang import language as T

from tile_kernels.config import get_amax_clamp_for_quant, is_ascend
from tile_kernels.quant.types import QuantTensor
from tile_kernels.utils import align, ceil_div, str_to_dtype
from tile_kernels.config import dtype_nbytes


@functools.lru_cache(maxsize=8)
def get_best_vectorize_size(dtype: T.dtype) -> int:
    major, _ = torch.cuda.get_device_capability()
    return (16 if major < 10 else 32) // dtype_nbytes(dtype)


@dataclass(frozen=True)
class BaseCastConfig:
    torch_dtype: torch.dtype = torch.float8_e4m3fn
    sf_block: tuple[int, int] = (1, 1)
    use_tma_aligned_col_major_sf: bool = False
    use_packed_ue8m0: bool = False
    use_e4m3_sf: bool = False

    @property
    def with_sf(self) -> bool:
        return self.torch_dtype in [torch.float8_e4m3fn, torch.int8, torch.uint32]

    @property
    def dtype(self) -> T.dtype:
        return T.dtype(self.torch_dtype) if self.torch_dtype != torch.int8 else T.float4_e2m1fn

    @property
    def sf_torch_dtype(self) -> torch.dtype:
        if self.use_packed_ue8m0:
            return torch.uint8
        elif self.use_e4m3_sf:
            return torch.float8_e4m3fn
        else:
            return torch.float32

    @property
    def sf_dtype(self) -> T.dtype:
        return T.dtype(self.sf_torch_dtype)

    @property
    def sf_col_pack(self) -> bool:
        return self.use_packed_ue8m0 and self.sf_block[1] == 1


@dataclass(frozen=True)
class CastInputConfig(BaseCastConfig):
    pass


@dataclass(frozen=True)
class CastOutputConfig(BaseCastConfig):
    round_sf: bool = False
    clamp_min_value: Optional[float] = None


def get_sf_inv_dtype(value_dtype: T.dtype, out_config: CastOutputConfig) -> T.dtype:
    if value_dtype == T.bfloat16 and out_config.round_sf:
        return T.bfloat16
    return T.float32


def get_packed_ue8m0_pack_factor() -> int:
    return 2 if is_ascend() else 4


def get_packed_ue8m0_torch_dtype() -> torch.dtype:
    return torch.int16 if is_ascend() else torch.int32


@functools.lru_cache
def get_cast_input_config_impl(
    dtype: torch.dtype,
    sf_block: Optional[tuple[int, int]] = None,
    use_tma_aligned_col_major_sf: bool = False,
    use_packed_ue8m0: bool = False,
    use_e4m3_sf: bool = False,
) -> CastInputConfig:
    if sf_block is None:
        assert dtype in (torch.bfloat16, torch.float32)
        return CastInputConfig(dtype)
    else:
        assert dtype in (torch.float8_e4m3fn, torch.int8, torch.uint8)
        return CastInputConfig(dtype, sf_block, use_tma_aligned_col_major_sf, use_packed_ue8m0, use_e4m3_sf)


def get_cast_input_and_config(
    x: Union[torch.Tensor, QuantTensor],
    sf_block: Optional[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor, CastInputConfig]:
    if isinstance(x, tuple):
        x, x_sf = x
        assert isinstance(sf_block, tuple) and isinstance(x, torch.Tensor) and isinstance(x_sf, torch.Tensor)
        assert x_sf.dtype in (torch.float32, torch.float8_e4m3fn, get_packed_ue8m0_torch_dtype())

        use_tma_aligned_col_major_sf = False
        use_packed_ue8m0 = False
        use_e4m3_sf = False
        if x_sf.stride(0) == 1 and x_sf.stride(1) != 1:
            use_tma_aligned_col_major_sf = True
            x_sf = x_sf.T
        else:
            assert x_sf.stride(1) == 1
        if x_sf.dtype == get_packed_ue8m0_torch_dtype():
            use_packed_ue8m0 = True
            x_sf = x_sf.view(torch.uint8)
        elif x_sf.dtype == torch.float8_e4m3fn:
            use_e4m3_sf = True

        config = get_cast_input_config_impl(x.dtype, sf_block, use_tma_aligned_col_major_sf, use_packed_ue8m0, use_e4m3_sf)
        assert not (config.use_e4m3_sf and is_ascend()), 'E4M3 SF is only supported on CUDA'
        return x, x_sf, config
    else:
        assert sf_block is None and isinstance(x, torch.Tensor)
        config = get_cast_input_config_impl(x.dtype)
        return x, None, config


@functools.lru_cache
def get_cast_output_config_impl(
    fmt: str,
    sf_block: tuple[int, int],
    use_tma_aligned_col_major_sf: bool,
    round_sf: bool,
    use_packed_ue8m0: bool,
    use_e4m3_sf: bool,
    clamp_min_value: Optional[float],
) -> CastOutputConfig:
    assert fmt in ('fp32', 'bf16', 'e4m3', 'e2m1')
    assert not use_packed_ue8m0 or round_sf
    assert not (round_sf and use_e4m3_sf)
    config = CastOutputConfig(
        torch_dtype=str_to_dtype(fmt),
        sf_block=sf_block,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
        use_e4m3_sf=use_e4m3_sf,
        clamp_min_value=clamp_min_value,
    )
    return config


def get_cast_output_config(
    fmt: str,
    sf_block: tuple[int, int],
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    use_e4m3_sf: bool = False,
) -> CastOutputConfig:
    if fmt in ('e4m3', 'e2m1'):
        clamp_min_value = get_amax_clamp_for_quant(fmt, use_e4m3_sf)
    else:
        clamp_min_value = None
    return get_cast_output_config_impl(
        fmt,
        sf_block,
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
        use_e4m3_sf,
        clamp_min_value,
    )


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

    if not config.sf_col_pack:
        if config.use_packed_ue8m0:
            num_block_k = ceil_div(num_block_k, get_packed_ue8m0_pack_factor())
        if config.use_tma_aligned_col_major_sf:
            num_block_m, num_block_k = num_block_k, num_block_m
        if config.use_packed_ue8m0:
            num_block_k *= get_packed_ue8m0_pack_factor()
    else:
        num_block_m = ceil_div(num_block_m, get_packed_ue8m0_pack_factor())
        num_block_k *= get_packed_ue8m0_pack_factor()

    return num_block_m, num_block_k


def alloc_scaling_factors(
    shape: tuple[int, int],
    out_config: BaseCastConfig,
    device: torch.device = 'cuda',
) -> torch.Tensor:
    """
    Allocate scaling factors for quantization.
    """
    sf_shape = get_sf_shape(shape, out_config)
    if out_config.use_tma_aligned_col_major_sf:
        # CUDA: col-major SF must be aligned to 16 bytes for TMA load; Ascend needs no alignment.
        aligned_sf_shape_1 = align(sf_shape[1], 16 // dtype_nbytes(out_config.sf_dtype)) if not is_ascend() else sf_shape[1]
        scaling_factor = torch.empty(
            size=(sf_shape[0], aligned_sf_shape_1),
            dtype=out_config.sf_torch_dtype,
            device=device,
        )
        # Torch gives unexpected stride when zero shape is placed at the last dimension
        if aligned_sf_shape_1 == 0:
            scaling_factor.as_strided_(scaling_factor.shape, (aligned_sf_shape_1, 1))
        return scaling_factor[:, : sf_shape[1]]
    else:
        return torch.empty(
            size=sf_shape,
            dtype=out_config.sf_torch_dtype,
            device=device,
        )


def cast_epilogue(out_sf: torch.Tensor, config: BaseCastConfig) -> torch.Tensor:
    """Post-process the sf-factor tensor after a cast kernel launch.

    Args:
        out_sf: Raw sf-factor tensor produced by the kernel.
        config: Cast configuration used during the kernel launch.

    Returns:
        Corrected sf-factor tensor with proper layout and shape.
    """
    # Make corrected SF tensor
    if config.use_packed_ue8m0:
        out_sf = out_sf.view(dtype=get_packed_ue8m0_torch_dtype())
    return out_sf.T if config.use_tma_aligned_col_major_sf else out_sf


@T.macro
def get_sf_and_inv(amax: float, out_config: CastOutputConfig, sf_inv_dtype: T.dtype = T.float32):
    # Clamp with min value
    clamped_amax = T.max(amax, out_config.clamp_min_value)

    max_value = T.max_value(out_config.dtype)
    sf = T.alloc_var(T.float32)
    if out_config.round_sf:
        # round_sf only uses ceil(log2(sf)); for supported formats this
        # reciprocal multiply preserves the same power-of-two bin as division.
        sf = clamped_amax * (T.float32(1) / max_value)
    else:
        sf = clamped_amax / max_value
        if out_config.use_e4m3_sf:
            sf_fp8 = T.cast(sf, T.float8_e4m3fn)
            return sf_fp8, 1 / T.cast(sf_fp8, T.float32)
        else:
            return sf, max_value / clamped_amax

    # Round into 2's power
    bits = T.reinterpret(sf, T.uint32)
    # clamp_min_value ensures sf is positive and normal before exponent extraction.
    # `(bits - 1) >> 23 + 1` gives ceil(log2).
    exp_sf = ((bits - 1) >> 23) + 1 - 127
    if sf_inv_dtype == T.bfloat16:
        sf_inv = T.reinterpret(T.uint16(127 - exp_sf) << 7, T.bfloat16)
    else:
        sf_inv = T.reinterpret((127 - exp_sf) << 23, T.float32)
    if out_config.use_packed_ue8m0:
        return T.uint8(exp_sf + 127), sf_inv
    else:
        return T.reinterpret((127 + exp_sf) << 23, T.float32), sf_inv


@T.macro
def get_sf_index(m_idx: int, k_idx: int, config: BaseCastConfig):
    pack_factor = get_packed_ue8m0_pack_factor()
    if config.sf_col_pack:
        return m_idx // pack_factor, k_idx * pack_factor + m_idx % pack_factor
    if config.use_packed_ue8m0 and config.use_tma_aligned_col_major_sf:
        return k_idx // pack_factor, m_idx * pack_factor + k_idx % pack_factor
    if config.use_tma_aligned_col_major_sf:
        return k_idx, m_idx
    return m_idx, k_idx


@T.macro
def load_sf(tensor: T.Tensor, m_idx: int, k_idx: int, config: BaseCastConfig):
    row, col = get_sf_index(m_idx, k_idx, config)
    return tensor[row, col]


@T.macro
def transform_sf(sf: Union[T.float32, T.float8_e4m3fn, T.uint8], config: BaseCastConfig) -> T.float32:
    if config.use_packed_ue8m0:
        return T.reinterpret(T.uint32(sf) << 23, T.float32)
    elif config.use_e4m3_sf:
        return T.cast(sf, T.float32)
    else:
        return sf


@T.macro
def store_sf(tensor: T.Tensor, sf: Union[T.float32, T.float8_e4m3fn, T.uint8], m_idx: int, k_idx: int, config: BaseCastConfig):
    row, col = get_sf_index(m_idx, k_idx, config)
    tensor[row, col] = sf


def unpack_from_e2m1fn_x2(x: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Decode a uint8/int8 tensor of packed fp4 values to out_dtype. Used mainly for debugging.

    The input tensor is expected to have shape [..., K] where each element contains
    two packed fp4 values.
    The output tensor will have shape [..., 2 * K] with the decoded values in out_dtype.
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

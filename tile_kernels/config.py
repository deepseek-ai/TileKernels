import functools
import operator
import torch
import os
from tilelang import language as T

_DEFAULT_NUM_SMS = 0
_DEFAULT_USE_PDL = False
_DEFAULT_DETERMINISTIC = False
_DEFAULT_TOKEN_ALIGNMENT = 128
_DEFAULT_CLAMP_MIN_VALUES = {
    ('e4m3', False): 1e-4,
    ('e4m3', True): float(T.max_value(T.float8_e4m3fn)) * (2**-9),
    ('e2m1', False): float(T.max_value(T.float4_e2m1fn) * (2**-126)),
    ('e2m1', True): float(T.max_value(T.float4_e2m1fn)) * (2**-9),
}

_num_sms = _DEFAULT_NUM_SMS
_use_pdl = _DEFAULT_USE_PDL
_deterministic_algorithms = _DEFAULT_DETERMINISTIC
_token_alignment = _DEFAULT_TOKEN_ALIGNMENT
_clamp_min_values = _DEFAULT_CLAMP_MIN_VALUES.copy()


@functools.lru_cache(maxsize=1)
def is_ascend() -> bool:
    return os.path.exists('/dev/davinci_manager')


@functools.lru_cache(maxsize=1)
def get_device() -> str:
    return 'npu' if is_ascend() else 'cuda'


def _validate_quant_fmt(fmt: str) -> None:
    if fmt not in ('e4m3', 'e2m1'):
        raise ValueError(f'Unsupported quant fmt {fmt}')


@functools.lru_cache(maxsize=None)
def get_device_num_sms() -> int:
    if is_ascend():
        prop = torch.npu.get_device_properties(torch.npu.current_device())
        return prop.cube_core_num
    else:
        prop = torch.cuda.get_device_properties(torch.cuda.current_device())
        return prop.multi_processor_count


def set_num_sms(num_sms: int) -> None:
    global _num_sms
    assert 0 < num_sms <= get_device_num_sms()
    _num_sms = num_sms


def get_num_sms() -> int:
    global _num_sms
    if _num_sms == 0:
        return get_device_num_sms()
    return _num_sms


# Pick the accessor that matches the kernel type:
#   - Mix kernel (AIC + AIV): use `get_num_ai_cores`
#   - pure cube kernel:       use `get_num_cube_cores`
#   - pure vector kernel:     use `get_num_vec_cores`
get_num_ai_cores = get_num_sms
get_num_cube_cores = get_num_sms


def get_num_vec_cores() -> int:
    return get_num_sms() * 2


@functools.lru_cache(maxsize=None)
def get_max_smem_per_sm() -> int:
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.shared_memory_per_multiprocessor


@functools.lru_cache(maxsize=None)
def get_max_ub_per_vector_core(use_simt: bool = True) -> int:
    assert is_ascend()
    return 216 * 1024 if use_simt else 248 * 1024


def set_token_alignment(token_alignment: int) -> None:
    """Set a positive integer token alignment, preserving the old value on failure."""
    global _token_alignment
    token_alignment = operator.index(token_alignment)
    if token_alignment <= 0:
        raise ValueError('token_alignment must be a positive integer')
    _token_alignment = token_alignment


def get_token_alignment() -> int:
    global _token_alignment
    return _token_alignment


def reset_runtime_config() -> None:
    global _num_sms, _use_pdl, _deterministic_algorithms, _token_alignment, _clamp_min_values
    _num_sms = _DEFAULT_NUM_SMS
    _use_pdl = _DEFAULT_USE_PDL
    _deterministic_algorithms = _DEFAULT_DETERMINISTIC
    _token_alignment = _DEFAULT_TOKEN_ALIGNMENT
    _clamp_min_values = _DEFAULT_CLAMP_MIN_VALUES.copy()


def set_pdl(use_pdl: bool) -> None:
    global _use_pdl
    _use_pdl = use_pdl


def get_pdl() -> bool:
    global _use_pdl
    return _use_pdl


def use_deterministic_algorithms(enabled: bool) -> None:
    global _deterministic_algorithms
    _deterministic_algorithms = bool(enabled)


def get_deterministic_algorithms() -> bool:
    global _deterministic_algorithms
    return _deterministic_algorithms


def set_amax_clamp_for_quant(fmt: str, use_e4m3_sf: bool, clamp_min_value: float) -> None:
    global _clamp_min_values
    _validate_quant_fmt(fmt)
    _clamp_min_values[(fmt, use_e4m3_sf)] = clamp_min_value


def get_amax_clamp_for_quant(fmt: str, use_e4m3_sf: bool) -> float:
    global _clamp_min_values
    _validate_quant_fmt(fmt)
    return _clamp_min_values[(fmt, use_e4m3_sf)]

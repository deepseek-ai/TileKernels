import functools
from typing import Optional

import torch

_num_sms = 0


@functools.lru_cache(maxsize=None)
def _get_device_num_sms(device_index: int) -> int:
    prop = torch.cuda.get_device_properties(device_index)
    return prop.multi_processor_count


@functools.lru_cache(maxsize=None)
def _get_max_smem_per_sm(device_index: int) -> int:
    prop = torch.cuda.get_device_properties(device_index)
    return prop.shared_memory_per_multiprocessor


def _resolve_device_index(device_index: Optional[int]) -> int:
    # Resolve before the cached call: caching under `None` would pin whichever device
    # happened to be current on the first call and reuse that value on every other device.
    if device_index is None:
        return torch.cuda.current_device()
    return device_index


def get_device_num_sms(device_index: Optional[int] = None) -> int:
    # The cached key is always a concrete device index, so a second device never reuses
    # the first device's value, whether or not the caller passed an index.
    return _get_device_num_sms(_resolve_device_index(device_index))


def set_num_sms(num_sms: int) -> None:
    # The override below is process-global, so the bound is checked against the current
    # device only; the value is not re-validated per device when it is later used.
    global _num_sms
    assert 0 < num_sms <= get_device_num_sms()
    _num_sms = num_sms


def get_num_sms(device_index: Optional[int] = None) -> int:
    # `_num_sms` set through `set_num_sms` is a process-global override; it wins
    # over the per-device value.
    global _num_sms
    if _num_sms == 0:
        return get_device_num_sms(device_index)
    return _num_sms


def get_max_smem_per_sm(device_index: Optional[int] = None) -> int:
    return _get_max_smem_per_sm(_resolve_device_index(device_index))

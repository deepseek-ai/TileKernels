import functools
import torch
from tilelang.utils.target import determine_target

_num_sms = 0


@functools.lru_cache(maxsize=None)
def get_warp_size() -> int:
    """Return the hardware wavefront/warp size for the current target.

    - CDNA GPUs (gfx9xx): wave64 → 64
    - RDNA GPUs (gfx10xx/11xx/12xx) and CUDA: wave32 → 32
    """
    target = determine_target(return_object=True)
    if target.kind.name == 'hip' and 'mcpu' in target.attrs:
        mcpu = str(target.attrs['mcpu'])
        if mcpu.startswith('gfx9'):  # CDNA family: gfx908, gfx90a, gfx94x, gfx950
            return 64
    return 32


@functools.lru_cache(maxsize=None)
def get_device_num_sms() -> int:
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


@functools.lru_cache(maxsize=None)
def get_max_smem_per_sm() -> int:
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.shared_memory_per_multiprocessor

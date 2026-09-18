import contextlib

import torch


def get_device_guard(device: torch.device):
    """Return a context manager that makes `device` the current CUDA device.

    Needed by the MoE entry points because output allocation, JIT compilation and
    kernel launch all resolve `cuda` against the current device, not against the
    device the input tensors already live on.
    """
    if device.type == 'cuda':
        return torch.cuda.device(device)
    return contextlib.nullcontext()


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0

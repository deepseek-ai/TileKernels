import os
import statistics
import sys

import torch


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def print_average_perf(latency_list: list[float], bandwidth_list: list[float], relative_speed_list: list[float]) -> None:
    if len(latency_list) == 0 or len(bandwidth_list) == 0:
        print('Empty latency_list and bandwidth_list')
        return
    print(f'Average Performance: {statistics.geometric_mean(latency_list):.1f} us, {statistics.geometric_mean(bandwidth_list):.0f} GB/s, '
          f'{statistics.geometric_mean(relative_speed_list) if len(relative_speed_list) > 0 else 1:.2f}x speedup')


def dtype_to_str(dtype: torch.dtype) -> str:
    mapping = {
        torch.float32: 'fp32',
        torch.bfloat16: 'bf16',
        torch.float8_e4m3fn: 'e4m3',
        torch.int8: 'e2m1', # int8 represents FP4 e2m1 format
    }

    if dtype not in mapping:
        raise ValueError(f'Unsupported dtype: {dtype}. Only fp32, bf16, e4m3, and int8(e2m1) are supported')

    return mapping[dtype]


def _format_value(value):
    if isinstance(value, torch.dtype):
        return dtype_to_str(value)
    if isinstance(value, tuple):
        return 'x'.join(str(v) for v in value)
    if isinstance(value, dict):
        return ','.join(f'{k}={_format_value(v)}' for k, v in sorted(value.items()))
    if value is None:
        return 'None'
    return str(value)


_SHORT_NAME = {
    'num_ep_ranks': 'ep',
    'num_experts': 'experts',
    'use_tma_aligned_col_major_sf': 'col',
    'use_packed_ue8m0': 'ue8m0',
    'round_sf': 'round'
}

_WIDTH = {
    'num_tokens': 5,
    'num_ep_ranks': 2,
    'num_experts': 3,
    'hidden': 4,
    'use_tma_aligned_col_major_sf': 1,
    'use_packed_ue8m0': 1,
    'round_sf': 1,
    'num_per_channels': 4,
}

def make_param_key(params: dict) -> str:
    """Generate a unique key for a benchmark record."""
    param_str = ','.join(f'{_SHORT_NAME.get(k, k)}={format(v, f">{_WIDTH.get(k)}") if k in _WIDTH else v}' for k, v in params.items() if v != None)
    return f'{param_str}'


def make_param_id(params: dict) -> str:
    parts = []

    for key in params:
        value = params[key]
        parts.append(f'{key}={_format_value(value)}')

    return '-'.join(parts) if parts else 'default'

import os

import pytest

from tile_kernels.config import get_device
from tile_kernels.engram import fused_weight
from tile_kernels.rand import randn
from tile_kernels.testing.bench import make_param_id
from tile_kernels.testing.generator import generate_hidden_sizes, get_test_level
from tile_kernels.testing.numeric import assert_equal, count_bytes
from tile_kernels.utils import str_to_dtype

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_test_data(params):
    hc_mult = params['hc']
    hidden_size = params['hidden']
    dtype = str_to_dtype(params['dtype'])
    device = get_device()
    wh_data = randn(hc_mult, hidden_size, dtype=dtype, device=device)
    we_data = randn(hc_mult, hidden_size, dtype=dtype, device=device)
    return (wh_data, we_data)


def generate_test_params(level: int) -> list[dict]:
    return [
        {'hc': hc, 'hidden': hidden_size, 'dtype': dtype}
        for hc in (4,)
        for hidden_size in generate_hidden_sizes(level, ['main'], align=128)
        for dtype in ['bf16', 'fp32']
    ]


@pytest.mark.parametrize('params', generate_test_params(max(get_test_level(), 1)), ids=make_param_id)
def test_engram_fused_weight(params):
    wh_data, we_data = generate_test_data(params)

    ref = wh_data.float() * we_data.float()
    out = fused_weight(wh_data, we_data)

    assert_equal(out, ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(1), ids=make_param_id)
def test_engram_fused_weight_benchmark(benchmark_timer, benchmark_record, params):
    wh_data, we_data = generate_test_data(params)
    out = fused_weight(wh_data, we_data)

    t_us = benchmark_timer(lambda: fused_weight(wh_data, we_data))

    num_bytes = count_bytes(wh_data, we_data, out)
    bandwidth_gbs = num_bytes / t_us / 1e3
    benchmark_record(
        kernel='fused_weight',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )

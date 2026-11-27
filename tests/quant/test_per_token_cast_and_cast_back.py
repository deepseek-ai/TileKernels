import os

import pytest

import tile_kernels
from tile_kernels.rand import randn
from tile_kernels.config import get_device, is_ascend
from tile_kernels.testing.bench import make_param_id
from tile_kernels.testing.generator import generate_hidden_sizes, generate_num_tokens, get_test_level
from tile_kernels.testing.numeric import assert_equal, check_bias, count_bytes
from tile_kernels.utils import str_to_dtype

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_test_data(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    in_dtype = params['in_dtype']

    return randn((num_tokens, hidden), dtype=str_to_dtype(in_dtype), device=get_device())


def generate_test_params(level: int) -> list[dict]:
    return [
        {
            'num_tokens': num_tokens,
            'hidden': hidden_size,
            'in_dtype': in_dtype,
            'num_per_channels': num_per_channels,
            'fmt': fmt,
            'round_sf': round_sf,
            'use_e4m3_sf': use_e4m3_sf,
        }
        for num_tokens in generate_num_tokens(level)
        for hidden_size in generate_hidden_sizes(level, ['head'], align=128 if is_ascend() else 64)
        for in_dtype in ('bf16', 'fp32')
        for num_per_channels in (16, 32)
        for fmt in ('e4m3', 'e2m1')
        for round_sf, use_e4m3_sf in (((False, True), (True, False)) if level == 0 else ((False, True), (True, False), (False, False)))
        if not is_ascend() or (num_per_channels, use_e4m3_sf) in ((16, True), (32, False))
        if hidden_size % num_per_channels == 0
    ]


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_per_token_cast_and_cast_back(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    in_dtype = params['in_dtype']
    num_per_channels = params['num_per_channels']
    fmt = params['fmt']
    round_sf = params['round_sf']
    use_e4m3_sf = params['use_e4m3_sf']

    original_x = generate_test_data(params)

    func = lambda x: tile_kernels.quant.per_token_cast_and_cast_back(
        x, fmt, num_per_channels=num_per_channels, round_sf=round_sf, use_e4m3_sf=use_e4m3_sf
    )

    x = original_x.clone()
    func(x)

    x_quant, x_sf = tile_kernels.torch.cast(original_x, fmt, (1, num_per_channels), round_sf=round_sf, use_e4m3_sf=use_e4m3_sf)
    x_ref = tile_kernels.torch.cast_back((x_quant, x_sf), in_dtype, (1, num_per_channels))

    assert_equal(x, x_ref)

    check_bias(x, original_x)

    if num_tokens > 0:
        x_non_contiguous = randn((num_tokens, hidden * 2), dtype=original_x.dtype, device=original_x.device)[:, :hidden]
        x_non_contiguous.copy_(original_x)
        func(x_non_contiguous)
        assert_equal(x_non_contiguous, x, check_stride=False)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0), ids=make_param_id)
def test_per_token_cast_and_cast_back_benchmark(benchmark_timer, benchmark_record, params):
    num_per_channels = params['num_per_channels']
    fmt = params['fmt']
    round_sf = params['round_sf']
    use_e4m3_sf = params['use_e4m3_sf']

    x = generate_test_data(params)
    func = lambda: tile_kernels.quant.per_token_cast_and_cast_back(
        x, fmt, num_per_channels=num_per_channels, round_sf=round_sf, use_e4m3_sf=use_e4m3_sf
    )

    t_us = benchmark_timer(func)
    num_bytes = 2 * count_bytes(x)

    benchmark_record(
        kernel='per_token_cast_and_cast_back',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )

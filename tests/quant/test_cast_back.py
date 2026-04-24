import os
import pytest
import torch

import tile_kernels
from tile_kernels.testing.bench import dtype_to_str, make_param_id
from tile_kernels.testing.generator import generate_hidden_sizes, generate_num_tokens
from tile_kernels.testing.numeric import assert_equal, calc_diff, count_bytes
from tests.conftest import IS_HIP

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_test_data_per_token(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    fmt = params['fmt']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']
    round_sf = params['round_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']
    num_per_channels = params['num_per_channels']
    out_dtype = params['out_dtype']

    x = torch.randn((num_tokens, hidden), dtype=out_dtype, device='cuda')
    x_fp8, x_sf = tile_kernels.quant.per_token_cast(
        x, fmt, num_per_channels=num_per_channels,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )
    out_dtype_str = dtype_to_str(out_dtype)
    func = lambda: tile_kernels.quant.per_token_cast_back((x_fp8, x_sf), out_dtype_str, num_per_channels=num_per_channels)

    return (x, x_fp8, x_sf, out_dtype_str, func)


def generate_test_data(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    round_sf = params['round_sf']
    fmt = params['fmt']
    out_dtype = params['out_dtype']
    num_per_tokens = params['num_per_tokens']
    num_per_channels = params['num_per_channels']

    x = torch.randn((num_tokens, hidden), dtype=out_dtype, device='cuda')
    x_casted, x_sf = tile_kernels.torch.cast(x, fmt, (num_per_tokens, num_per_channels), round_sf=round_sf)
    out_dtype_str = dtype_to_str(out_dtype)
    func = lambda: tile_kernels.quant.cast_back(
        (x_casted, x_sf), out_dtype_str, (num_per_tokens, num_per_channels)
    )

    return (x, x_casted, x_sf, out_dtype_str, func)


def generate_test_params_per_token(is_benchmark: bool) -> list[dict]:
    return [
        {
            'num_tokens': num_tokens,
            'hidden': hidden_size,
            'fmt': fmt,
            'use_tma_aligned_col_major_sf': use_tma_aligned_col_major_sf,
            'round_sf': round_sf,
            'use_packed_ue8m0': use_packed_ue8m0,
            'num_per_channels': num_per_channels,
            'out_dtype': out_dtype,
        }
        for num_tokens in generate_num_tokens(is_benchmark=is_benchmark)
        for hidden_size in generate_hidden_sizes()
        for fmt in (('e4m3',) if IS_HIP else ('e2m1', 'e4m3'))
        for use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0 in (
            [(False, True, False)] if IS_HIP else [(False, True, False), (True, True, True)]
        )
        for num_per_channels in (128, hidden_size)
        for out_dtype in (torch.float32, torch.bfloat16)
    ]


def generate_test_params(is_benchmark: bool) -> list[dict]:
    return [
        {
            'num_tokens': num_tokens,
            'hidden': hidden_size,
            'round_sf': round_sf,
            'fmt': fmt,
            'out_dtype': out_dtype,
            'num_per_tokens': num_per_tokens,
            'num_per_channels': num_per_channels,
        }
        for num_tokens in generate_num_tokens(is_benchmark=is_benchmark)
        for hidden_size in generate_hidden_sizes()
        for round_sf in (False, True)
        for fmt in ('e4m3',)
        for out_dtype in (torch.bfloat16, torch.float32)
        for num_per_tokens, num_per_channels in ((128, 1), (128, 128))
    ]


@pytest.mark.parametrize('params', generate_test_params_per_token(is_benchmark=False), ids=make_param_id)
def test_cast_back_per_token(params):
    hidden = params['hidden']
    fmt = params['fmt']
    num_per_channels = params['num_per_channels']

    # Test correctness
    x, x_fp8, x_sf, out_dtype_str, func = generate_test_data_per_token(params)
    x_fp8_bf16 = func()
    x_fp8_bf16_ref = tile_kernels.torch.cast_back((x_fp8, x_sf), out_dtype_str, (1, num_per_channels))

    diff = calc_diff(x, x_fp8_bf16)
    assert diff < (2e-2 if fmt == 'e2m1' else 1e-3), f'{x}, {x_fp8_bf16}, {fmt=}, {hidden=}, {num_per_channels=}, {diff=}'

    assert_equal(x_fp8_bf16, x_fp8_bf16_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params_per_token(is_benchmark=True), ids=make_param_id)
def test_cast_back_per_token_benchmark(benchmark_timer, benchmark_record, params):
    x, x_fp8, x_sf, out_dtype_str, func = generate_test_data_per_token(params)

    t_us = benchmark_timer(func)
    num_bytes = count_bytes(x, x_fp8, x_sf)

    benchmark_record(
        kernel='cast_back_per_token',
        operation='fwd',
        params={**params, 'out_dtype': out_dtype_str},
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_cast_back(params):
    num_per_tokens = params['num_per_tokens']
    num_per_channels = params['num_per_channels']

    _, x_casted, x_sf, out_dtype_str, func = generate_test_data(params)
    x_casted_back = func()
    x_casted_back_ref = tile_kernels.torch.cast_back((x_casted, x_sf), out_dtype_str, (num_per_tokens, num_per_channels))

    assert_equal(x_casted_back, x_casted_back_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_cast_back_benchmark(benchmark_timer, benchmark_record, params):
    x, x_casted, x_sf, out_dtype_str, func = generate_test_data(params)

    t_us = benchmark_timer(func)
    num_bytes = count_bytes(x, x_casted, x_sf)

    benchmark_record(
        kernel='cast_back',
        operation='fwd',
        params={**params, 'out_dtype': out_dtype_str},
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )

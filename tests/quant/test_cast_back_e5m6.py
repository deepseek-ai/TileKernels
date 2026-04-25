import os
import pytest
import torch

import tile_kernels
from tile_kernels.testing.bench import dtype_to_str, make_param_id
from tile_kernels.testing.generator import generate_hidden_sizes, generate_num_tokens, generate_e5m6_inputs
from tile_kernels.testing.numeric import assert_equal, calc_diff, count_bytes
from tile_kernels.torch import cast_back_from_e5m6
from tests.conftest import IS_HIP

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_test_data(params):
    x = params['x']
    hidden = params['hidden']
    num_per_channels = params['num_per_channels']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']
    round_sf = params['round_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']
    out_dtype = params['out_dtype']

    x_e5m6, x_sf = tile_kernels.quant.per_token_cast(
        x, 'e5m6', num_per_channels=num_per_channels,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )
    out_dtype_str = dtype_to_str(out_dtype)
    func = lambda: tile_kernels.quant.cast_back((x_e5m6, x_sf), out_dtype_str, (1, hidden), x_special_fmt='e5m6')
    torch_ref_func = lambda: cast_back_from_e5m6((x_e5m6, x_sf), out_dtype_str, (1, hidden))

    return (x_e5m6, x_sf, out_dtype_str, func, torch_ref_func)


def generate_test_params(is_benchmark: bool) -> list[dict]:
    return [
        {
            'num_tokens': num_tokens,
            'hidden': hidden_size,
            'num_per_channels': num_per_channels,
            'use_tma_aligned_col_major_sf': use_tma_aligned_col_major_sf,
            'round_sf': round_sf,
            'use_packed_ue8m0': use_packed_ue8m0,
            'out_dtype': out_dtype,
        }
        for num_tokens in generate_num_tokens(is_benchmark=is_benchmark)
        for hidden_size in generate_hidden_sizes()
        for num_per_channels in (hidden_size, )
        for use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0 in (
            [(False, True, False)] if IS_HIP else [(False, True, False), (True, True, True)]
        )
        for out_dtype in (torch.bfloat16, torch.float32)
    ]


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_cast_back_e5m6(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']
    round_sf = params['round_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']
    out_dtype = params['out_dtype']
    num_per_channels = params['num_per_channels']

    for x, is_special in generate_e5m6_inputs(num_tokens, hidden, out_dtype):
        x_e5m6, x_sf, out_dtype_str, func, torch_ref_func = generate_test_data({
            'x': x,
            'hidden': hidden,
            'num_per_channels': num_per_channels,
            'use_tma_aligned_col_major_sf': use_tma_aligned_col_major_sf,
            'round_sf': round_sf,
            'use_packed_ue8m0': use_packed_ue8m0,
            'out_dtype': out_dtype,
        })
        x_back = func()

        # Check accuracy vs original input
        diff = calc_diff(x, x_back)
        threshold = 5e-6 if is_special else 1e-4
        assert diff < threshold, f'{hidden=}, {round_sf=}, {out_dtype_str=}, {diff=}'

        # Check against torch/cast reference (always runs)
        x_back_ref = torch_ref_func()
        assert_equal(x_back, x_back_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_cast_back_e5m6_benchmark(benchmark_timer, benchmark_record, params):
    hidden = params['hidden']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']
    out_dtype = params['out_dtype']

    num_per_channels = hidden
    for x, is_special in generate_e5m6_inputs(params['num_tokens'], hidden, out_dtype):
        if is_special:
            continue

        x_e5m6, x_sf, out_dtype_str, func, torch_ref_func = generate_test_data({
            'x': x,
            'hidden': hidden,
            'num_per_channels': num_per_channels,
            'use_tma_aligned_col_major_sf': use_tma_aligned_col_major_sf,
            'round_sf': params['round_sf'],
            'use_packed_ue8m0': params['use_packed_ue8m0'],
            'out_dtype': out_dtype,
        })
        x_back = func()

        t_us = benchmark_timer(func)
        num_bytes = count_bytes(x_e5m6, x_sf, x_back)

        benchmark_record(
            kernel='cast_back_e5m6',
            operation='fwd',
            params={**params, 'num_per_channels': num_per_channels, 'out_dtype': out_dtype_str},
            time_us=t_us,
            bandwidth_gbs=num_bytes / t_us / 1e3,
        )

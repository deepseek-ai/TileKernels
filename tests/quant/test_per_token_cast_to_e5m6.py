import os
import pytest
import torch

import tile_kernels
from tile_kernels.testing import clear_unused_sf
from tile_kernels.testing.bench import dtype_to_str, make_param_id
from tile_kernels.testing.numeric import assert_equal, count_bytes
from tile_kernels.testing.generator import generate_hidden_sizes, generate_num_tokens, generate_e5m6_inputs
from tile_kernels.torch import cast_to_e5m6
from tests.conftest import IS_HIP

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_test_data(params):
    x = params['x']
    num_per_channels = params['num_per_channels']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']
    round_sf = params['round_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']

    func = lambda: tile_kernels.quant.per_token_cast(
        x, 'e5m6', num_per_channels=num_per_channels,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )
    torch_func_ref = lambda: cast_to_e5m6(
        x, num_per_channels=num_per_channels,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )
    return (func, torch_func_ref)


def generate_test_params(is_benchmark: bool) -> list[dict]:
    return [
        {
            'num_tokens': num_tokens,
            'hidden': hidden_size,
            'use_tma_aligned_col_major_sf': use_tma_aligned_col_major_sf,
            'round_sf': round_sf,
            'use_packed_ue8m0': use_packed_ue8m0,
            'in_dtype': in_dtype,
        }
        for num_tokens in generate_num_tokens(is_benchmark=is_benchmark)
        for hidden_size in generate_hidden_sizes()
        for use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0 in (
            [(False, True, False)] if IS_HIP else [(False, True, False), (True, True, True)]
        )
        for in_dtype in (torch.bfloat16, torch.float32)
    ]


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_per_token_cast_to_e5m6(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']
    round_sf = params['round_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']
    in_dtype = params['in_dtype']

    num_per_channels = hidden
    for x, is_special in generate_e5m6_inputs(num_tokens, hidden, in_dtype):
        # Test correctness
        func, torch_func_ref = generate_test_data({
            'x': x,
            'num_per_channels': num_per_channels,
            'use_tma_aligned_col_major_sf': use_tma_aligned_col_major_sf,
            'round_sf': round_sf,
            'use_packed_ue8m0': use_packed_ue8m0,
        })

        x_fp8, x_sf = func()
        x_fp8_ref, x_sf_ref = torch_func_ref()

        if use_packed_ue8m0:
            x_sf = clear_unused_sf(x_sf, hidden, num_per_channels)
            x_sf_ref = clear_unused_sf(x_sf_ref, hidden, num_per_channels)

        assert_equal(x_fp8, x_fp8_ref)
        assert_equal(x_sf, x_sf_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_per_token_cast_to_e5m6_benchmark(benchmark_timer, benchmark_record, params):
    hidden = params['hidden']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']
    in_dtype = params['in_dtype']

    num_per_channels = hidden
    for x, is_special in generate_e5m6_inputs(params['num_tokens'], hidden, in_dtype):
        if is_special:
            continue

        func, torch_func_ref = generate_test_data({
            'x': x,
            'num_per_channels': num_per_channels,
            'use_tma_aligned_col_major_sf': use_tma_aligned_col_major_sf,
            'round_sf': params['round_sf'],
            'use_packed_ue8m0': use_packed_ue8m0,
        })

        x_fp8, x_sf = func()

        t_us = benchmark_timer(func)
        num_bytes = count_bytes(x, x_fp8, x_sf)

        benchmark_record(
            kernel='per_token_cast_to_e5m6',
            operation='fwd',
            params={**params, 'num_per_channels': num_per_channels, 'in_dtype': dtype_to_str(in_dtype)},
            time_us=t_us,
            bandwidth_gbs=num_bytes / t_us / 1e3,
        )

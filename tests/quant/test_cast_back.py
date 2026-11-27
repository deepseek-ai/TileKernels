import os
import pytest

import tile_kernels
from tile_kernels.rand import randn
from tile_kernels.testing.bench import make_param_id
from tile_kernels.utils import str_to_dtype
from tile_kernels.testing.generator import generate_cast_config, generate_hidden_sizes, generate_num_tokens, get_test_level
from tile_kernels.testing.numeric import assert_equal, calc_diff, count_bytes
from tile_kernels.config import get_device, is_ascend

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_test_data_per_token(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    fmt = params['fmt']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']
    round_sf = params['round_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']
    use_e4m3_sf = params['use_e4m3_sf']
    num_per_channels = params['num_per_channels']
    out_dtype = params['out_dtype']

    x = randn((num_tokens, hidden), dtype=str_to_dtype(out_dtype), device=get_device())
    x_fp8, x_sf = tile_kernels.quant.per_token_cast(
        x,
        fmt,
        num_per_channels=num_per_channels,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
        use_e4m3_sf=use_e4m3_sf,
    )
    func = lambda: tile_kernels.quant.per_token_cast_back((x_fp8, x_sf), out_dtype, num_per_channels=num_per_channels)

    return (x, x_fp8, x_sf, out_dtype, func)


def generate_test_data(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    round_sf = params['round_sf']
    fmt = params['fmt']
    out_dtype = params['out_dtype']
    num_per_tokens = params['num_per_tokens']
    num_per_channels = params['num_per_channels']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']

    x = randn((num_tokens, hidden), dtype=str_to_dtype(out_dtype), device=get_device())
    x_casted, x_sf = tile_kernels.torch.cast(
        x,
        fmt,
        (num_per_tokens, num_per_channels),
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )
    func = lambda: tile_kernels.quant.cast_back((x_casted, x_sf), out_dtype, (num_per_tokens, num_per_channels))

    return (x, x_casted, x_sf, out_dtype, func)


def generate_test_params_per_token(level: int) -> list[dict]:
    return [
        {
            'num_tokens': num_tokens,
            'hidden': hidden_size,
            'fmt': fmt,
            'use_tma_aligned_col_major_sf': use_tma_aligned_col_major_sf,
            'round_sf': round_sf,
            'use_packed_ue8m0': use_packed_ue8m0,
            'use_e4m3_sf': use_e4m3_sf,
            'num_per_channels': num_per_channels,
            'out_dtype': out_dtype,
        }
        for num_tokens in generate_num_tokens(level)
        for hidden_size in generate_hidden_sizes(level, ['main', 'ffn', '2*ffn', 'q_lora', '128*indexer', 'wv', 'engram_head'], align=128)
        for fmt in ('e2m1', 'e4m3')
        for use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0 in generate_cast_config(level, fmt)
        for num_per_channels in ((32,) if is_ascend() else (16, 32))
        for use_e4m3_sf in ((False,) if (round_sf or is_ascend()) else (False, True))
        for out_dtype in ('fp32', 'bf16')
    ]


def generate_test_params(level: int) -> list[dict]:
    return [
        {
            'num_tokens': num_tokens,
            'hidden': hidden_size,
            'use_tma_aligned_col_major_sf': use_tma_aligned_col_major_sf and num_per_channels != 1,
            'round_sf': round_sf,
            'use_packed_ue8m0': use_packed_ue8m0,
            'fmt': fmt,
            'out_dtype': out_dtype,
            'num_per_tokens': num_per_tokens,
            'num_per_channels': num_per_channels,
        }
        for num_tokens in generate_num_tokens(level)
        for hidden_size in generate_hidden_sizes(level, ['main', 'ffn', '2*ffn', 'q_lora', '128*indexer', 'wv_input', 'wv', 'engram_input'], align=64)
        for fmt in ('e2m1', 'e4m3')
        for use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0 in generate_cast_config(level, fmt)
        for out_dtype in ('bf16', 'fp32')
        for num_per_tokens, num_per_channels in ((32, 1), (32, 32))
    ]


@pytest.mark.parametrize('params', generate_test_params_per_token(get_test_level()), ids=make_param_id)
def test_cast_back_per_token(params):
    hidden = params['hidden']
    fmt = params['fmt']
    use_e4m3_sf = params['use_e4m3_sf']
    num_per_channels = params['num_per_channels']

    # Test correctness
    x, x_fp8, x_sf, out_dtype_str, func = generate_test_data_per_token(params)
    x_fp8_bf16 = func()
    x_fp8_bf16_ref = tile_kernels.torch.cast_back((x_fp8, x_sf), out_dtype_str, (1, num_per_channels))

    diff = calc_diff(x, x_fp8_bf16)
    assert diff < (2e-2 if fmt == 'e2m1' or use_e4m3_sf else 1e-3), f'{x}, {x_fp8_bf16}, {fmt=}, {hidden=}, {num_per_channels=}, {diff=}'

    assert_equal(x_fp8_bf16, x_fp8_bf16_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params_per_token(0), ids=make_param_id)
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


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_cast_back(params):
    num_per_tokens = params['num_per_tokens']
    num_per_channels = params['num_per_channels']

    _, x_casted, x_sf, out_dtype_str, func = generate_test_data(params)
    x_casted_back = func()
    x_casted_back_ref = tile_kernels.torch.cast_back((x_casted, x_sf), out_dtype_str, (num_per_tokens, num_per_channels))

    assert_equal(x_casted_back, x_casted_back_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0), ids=make_param_id)
def test_cast_back_benchmark(benchmark_timer, benchmark_record, params):
    x, x_casted, x_sf, out_dtype_str, func = generate_test_data(params)

    t_us = benchmark_timer(func)
    num_bytes = count_bytes(x, x_casted, x_sf)

    benchmark_record(
        kernel='cast_back',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )

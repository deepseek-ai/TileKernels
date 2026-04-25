import os
import pytest
import torch


import tile_kernels
from tile_kernels.testing.numeric import assert_equal, count_bytes
from tile_kernels.testing.generator import generate_hidden_sizes, generate_num_tokens, generate_rand_float
from tile_kernels.testing.bench import make_param_id
from tests.conftest import IS_HIP

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def clamp_abs_ratio(t: torch.Tensor, max_ratio: float = 2**9):
    if t.numel() == 0:
        return t
    floor_val = t.abs().max() / max_ratio
    t = torch.sign(t) * torch.max(t.abs(), floor_val)
    return t


def generate_test_data(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    in_use_tma_aligned_col_major_sf = params['in_use_tma_aligned_col_major_sf']
    in_round_sf = params['in_round_sf']
    in_use_packed_ue8m0 = params['in_use_packed_ue8m0']
    out_use_tma_aligned_col_major_sf = params['out_use_tma_aligned_col_major_sf']
    out_round_sf = params['out_round_sf']
    out_use_packed_ue8m0 = params['out_use_packed_ue8m0']
    in_sf_block_m = params['in_sf_block'][0]
    in_sf_block_k = params['in_sf_block'][1]
    out_sf_block_m = params['out_sf_block'][0]
    out_sf_block_k = params['out_sf_block'][1]

    x = generate_rand_float((num_tokens, hidden))
    x = clamp_abs_ratio(x)
    x_fp4 = tile_kernels.torch.cast(
        x, 'e2m1', (in_sf_block_m, in_sf_block_k),
        use_tma_aligned_col_major_sf=in_use_tma_aligned_col_major_sf,
        round_sf=in_round_sf,
        use_packed_ue8m0=in_use_packed_ue8m0,
    )
    cast_func = lambda: tile_kernels.quant.per_block_cast_lossless(
        x_fp4, 'e4m3',
        x_block_size=(in_sf_block_m, in_sf_block_k),
        out_block_size=(out_sf_block_m, out_sf_block_k),
        use_tma_aligned_col_major_sf=out_use_tma_aligned_col_major_sf,
        round_sf=out_round_sf,
        use_packed_ue8m0=out_use_packed_ue8m0,
    )

    return (x, x_fp4, cast_func)


def generate_test_params(is_benchmark: bool) -> list[dict]:
    params = [
        {
            'num_tokens': num_tokens,
            'hidden': hidden_size,
            'in_use_tma_aligned_col_major_sf': in_use_tma_aligned_col_major_sf,
            'in_round_sf': in_round_sf,
            'in_use_packed_ue8m0': in_use_packed_ue8m0,
            'out_use_tma_aligned_col_major_sf': out_use_tma_aligned_col_major_sf,
            'out_round_sf': out_round_sf,
            'out_use_packed_ue8m0': out_use_packed_ue8m0,
            'out_sf_block': (out_sf_block_m, out_sf_block_k),
            'in_sf_block': (in_sf_block_m, in_sf_block_k),
        }
        for num_tokens in generate_num_tokens(is_benchmark=is_benchmark)
        for hidden_size in generate_hidden_sizes()
        for in_use_tma_aligned_col_major_sf, in_round_sf, in_use_packed_ue8m0 in [(False, True, False), (True, True, True)]
        for out_use_tma_aligned_col_major_sf, out_round_sf, out_use_packed_ue8m0 in [(False, True, False), (True, True, True)]
        for out_sf_block_m, out_sf_block_k in ((1, 128), (32, 32), (128, 128))
        for in_sf_block_m, in_sf_block_k in ((1, 32),)
        if out_sf_block_m % in_sf_block_m == 0 and out_sf_block_k % in_sf_block_k == 0
    ]
    return params


@pytest.mark.skipif(IS_HIP, reason='per_block_cast_lossless uses e2m1 (FP4) which is not supported on HIP/AMD targets')
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_per_block_cast_lossless(params):
    out_sf_block = params['out_sf_block']
    in_sf_block = params['in_sf_block']

    out_sf_block_m, out_sf_block_k = out_sf_block
    in_sf_block_m, in_sf_block_k = in_sf_block

    # Test Correctness
    _, x_fp4, cast_func = generate_test_data(params)
    x_fp8 = cast_func()

    x_fp8_fp32_ref = tile_kernels.torch.cast_back(x_fp4, 'fp32', (in_sf_block_m, in_sf_block_k))
    x_fp8_fp32 = tile_kernels.torch.cast_back(x_fp8, 'fp32', (out_sf_block_m, out_sf_block_k))
    assert_equal(x_fp8_fp32, x_fp8_fp32_ref)


@pytest.mark.benchmark
@pytest.mark.skipif(IS_HIP, reason='per_block_cast_lossless uses e2m1 (FP4) which is not supported on HIP/AMD targets')
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_per_block_cast_lossless_benchmark(benchmark_timer, benchmark_record, params):
    _, x_fp4, cast_func = generate_test_data(params)

    x_fp8 = cast_func()

    t_us = benchmark_timer(cast_func)
    num_bytes = count_bytes(x_fp4, x_fp8)
    benchmark_record(
        kernel='per_block_cast_lossless',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )

import os
import pytest
import torch

import tile_kernels
from tile_kernels.testing.bench import dtype_to_str, make_param_id
from tile_kernels.testing.numeric import assert_equal, count_bytes, check_bias
from tile_kernels.testing.generator import generate_hidden_sizes, generate_num_tokens
from tile_kernels.testing.quant import clear_unused_sf
from tests.conftest import IS_HIP

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_test_data(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    in_dtype = params['in_dtype']
    fmt = params['fmt']
    block_size = params['block_size']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']
    round_sf = params['round_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']

    x = torch.randn((num_tokens, hidden), dtype=in_dtype, device='cuda')
    base_args = dict(
        x=x, fmt=fmt, block_size=block_size,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )

    return (x, base_args)


def generate_test_params(is_benchmark: bool) -> list[dict]:
    return [
        {
            'num_tokens': num_tokens,
            'hidden': hidden_size,
            'in_dtype': in_dtype,
            'fmt': fmt,
            'use_tma_aligned_col_major_sf': use_tma_aligned_col_major_sf,
            'round_sf': round_sf,
            'use_packed_ue8m0': use_packed_ue8m0,
            'block_size': block_size,
        }
        for num_tokens in generate_num_tokens(is_benchmark=is_benchmark)
        for hidden_size in generate_hidden_sizes()
        for in_dtype in (torch.bfloat16, torch.float32)
        for fmt in (('e4m3',) if IS_HIP else ('e4m3', 'e2m1'))
        for use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0 in (
            [(False, True, False)] if IS_HIP else [(False, True, False), (True, True, True)]
        )
        for block_size in ((128, 128), (32, 32))
    ]


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_per_block_cast(params):
    hidden = params['hidden']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']
    block_size = params['block_size']

    x, base_args = generate_test_data(params)

    # Test cast
    x_casted, per_block_sf_inv = tile_kernels.quant.per_block_cast(**base_args)
    x_casted_ref, per_block_sf_inv_ref = tile_kernels.torch.cast(**base_args)
    x_casted_back = tile_kernels.torch.cast_back((x_casted, per_block_sf_inv), 'fp32', block_size)
    if use_packed_ue8m0:
        per_block_sf_inv = clear_unused_sf(per_block_sf_inv, hidden, block_size[1])
        per_block_sf_inv_ref = clear_unused_sf(per_block_sf_inv_ref, hidden, block_size[1])
    assert_equal(per_block_sf_inv, per_block_sf_inv_ref)
    assert_equal(x_casted, x_casted_ref)

    # Check bias
    check_bias(x_casted_back, x)

    # Test cast only mode
    if not use_tma_aligned_col_major_sf:
        # TMA aligned or packed ue8m0 sf is used for FP8/FP4 GEMM, not for other cast
        x_casted = tile_kernels.quant.per_block_cast_with_precomputed_sf(**base_args, sf=per_block_sf_inv)
        x_casted_ref = tile_kernels.torch.cast(**base_args, sf=per_block_sf_inv)
        assert_equal(x_casted, x_casted_ref)

    # Test sf only mode
    twice_per_block_sf_inv = tile_kernels.quant.per_block_cast_with_sf_only(**base_args)
    if use_packed_ue8m0:
        twice_per_block_sf_inv = clear_unused_sf(twice_per_block_sf_inv, hidden, block_size[1])
    assert_equal(twice_per_block_sf_inv, per_block_sf_inv_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_per_block_cast_benchmark(benchmark_timer, benchmark_record, params):
    x, args = generate_test_data(params)

    x_casted, per_block_sf_inv = tile_kernels.quant.per_block_cast(**args)

    t_us = benchmark_timer(lambda: tile_kernels.quant.per_block_cast(**args))
    num_bytes = count_bytes(x, x_casted, per_block_sf_inv)

    params['in_dtype'] = dtype_to_str(params['in_dtype'])
    benchmark_record(
        kernel='per_block_cast',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )

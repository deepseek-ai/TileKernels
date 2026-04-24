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
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']
    round_sf = params['round_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']
    x_block_size = params.get('x_block_size')

    in_with_sf_factor = in_dtype in (torch.float8_e4m3fn, torch.int8)

    if in_with_sf_factor:
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        original_x = x
        in_fmt = 'e4m3' if in_dtype == torch.float8_e4m3fn else 'e2m1'
        x = tile_kernels.torch.cast(
            x, in_fmt, x_block_size,
            use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
            round_sf=round_sf,
            use_packed_ue8m0=use_packed_ue8m0,
        )
    else:
        x = torch.randn((num_tokens, hidden), dtype=in_dtype, device='cuda')
        original_x = x

    base_args = dict(
        x=x, fmt=fmt, x_block_size=x_block_size,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )

    return (x, original_x, base_args, in_with_sf_factor)


def generate_test_params(is_benchmark: bool) -> list[dict]:
    params = [
        {
            'num_tokens': num_tokens,
            'hidden': hidden_size,
            'use_tma_aligned_col_major_sf': use_tma_aligned_col_major_sf,
            'round_sf': round_sf,
            'use_packed_ue8m0': use_packed_ue8m0,
            'in_dtype': in_dtype,
            'num_per_channels': num_per_channels,
            'x_block_size': x_block_size,
            'fmt': fmt,
        }
        for num_tokens in generate_num_tokens(is_benchmark=is_benchmark)
        for hidden_size in generate_hidden_sizes()
        for use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0 in (
            [(False, True, False)] if IS_HIP else [(False, True, False), (True, True, True)]
        )
        for in_dtype in (torch.float32, torch.bfloat16, torch.float8_e4m3fn) + (() if IS_HIP else (torch.int8,))
        for num_per_channels in ((32, 128) if in_dtype in (torch.float8_e4m3fn, torch.int8) else (32, 64, 128, hidden_size))
        for x_block_size in (((128, 128), (32, 32)) if in_dtype in (torch.float8_e4m3fn, torch.int8) else (None,))
        for fmt in (('e4m3',) if IS_HIP else ('e4m3', 'e2m1'))
    ]
    if is_benchmark:
        params = [p for p in params if p['use_packed_ue8m0']]
    return params


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_per_token_cast(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']
    round_sf = params['round_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']
    in_dtype = params['in_dtype']
    num_per_channels = params['num_per_channels']
    x_block_size = params.get('x_block_size')
    fmt = params['fmt']

    in_with_sf_factor = in_dtype in (torch.float8_e4m3fn, torch.int8)
    # Test correctness
    x, original_x, base_args, in_with_sf_factor = generate_test_data(params)
    func = lambda: tile_kernels.quant.per_token_cast(
        **base_args,
        num_per_channels=num_per_channels,
    )
    func_ref = lambda: tile_kernels.torch.cast(
        **base_args,
        block_size=(1, num_per_channels),
    )
    x_casted, x_sf = func()
    x_casted_ref, x_sf_ref = func_ref()
    x_casted_back = tile_kernels.torch.cast_back((x_casted, x_sf), 'fp32', (1, num_per_channels))

    if use_packed_ue8m0:
        x_sf = clear_unused_sf(x_sf, hidden, num_per_channels)
        x_sf_ref = clear_unused_sf(x_sf_ref, hidden, num_per_channels)

    assert_equal(x_casted, x_casted_ref)
    assert_equal(x_sf, x_sf_ref)

    # Check bias
    check_bias(x_casted_back, original_x)

    # Test non-contiguous input (stride(0) != hidden)
    if not in_with_sf_factor and fmt == 'e4m3' and not use_packed_ue8m0 and num_tokens > 0:
        x_non_contiguous = torch.randn((num_tokens, hidden * 2), dtype=in_dtype, device='cuda')[:, :hidden]
        x_non_contiguous.copy_(original_x)
        x_non_contiguous_casted, x_non_contiguous_sf = tile_kernels.quant.per_token_cast(
            x=x_non_contiguous, fmt=fmt, x_block_size=x_block_size,
            use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
            round_sf=round_sf, use_packed_ue8m0=use_packed_ue8m0,
            num_per_channels=num_per_channels,
        )
        assert_equal(x_non_contiguous_casted, x_casted)
        assert_equal(x_non_contiguous_sf, x_sf)

    if not in_with_sf_factor and num_per_channels != hidden:
        # Test cast only mode
        if not use_tma_aligned_col_major_sf:
            # TMA aligned or packed ue8m0 sf is used for FP8/FP4 GEMM, not for other cast
            x_casted = tile_kernels.quant.per_token_cast_with_precomputed_sf(
                **base_args, num_per_channels=num_per_channels, sf=x_sf
            )
            x_casted_ref = tile_kernels.torch.cast(**base_args, block_size=(1, num_per_channels), sf=x_sf)
            assert_equal(x_casted, x_casted_ref)

        # Test sf only mode
        twice_sf_inv = tile_kernels.quant.per_token_cast_with_sf_only(**base_args, num_per_channels=num_per_channels)
        if use_packed_ue8m0:
            twice_sf_inv = clear_unused_sf(twice_sf_inv, hidden, num_per_channels)
        assert_equal(twice_sf_inv, x_sf)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_per_token_cast_benchmark(benchmark_timer, benchmark_record, params):
    in_dtype = params['in_dtype']
    num_per_channels = params['num_per_channels']

    x, _, base_args, _ = generate_test_data(params)
    func = lambda: tile_kernels.quant.per_token_cast(
        **base_args,
        num_per_channels=num_per_channels,
    )
    x_casted, x_sf = func()

    t_us = benchmark_timer(func)
    num_bytes = count_bytes(x, x_casted, x_sf)

    benchmark_record(
        kernel='per_token_cast',
        operation='fwd',
        params={**params, 'in_dtype': dtype_to_str(in_dtype)},
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )

import os
import random
import pytest
import torch

import tile_kernels
from tile_kernels.testing.bench import make_param_id
from tile_kernels.utils import str_to_dtype
from tile_kernels.testing.generator import generate_hidden_sizes, generate_num_tokens, get_test_level, generate_cast_config, generate_rand_float
from tile_kernels.testing.numeric import assert_equal, count_bytes, check_bias
from tile_kernels.testing.quant import clear_unused_sf_col_pack
from tile_kernels.config import is_ascend

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_test_data(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    in_dtype = params['in_dtype']
    num_per_channels = params['num_per_channels']
    in_use_tma_aligned_col_major_sf = params['in_use_tma_aligned_col_major_sf']
    in_round_sf = params['in_round_sf']
    in_use_packed_ue8m0 = params['in_use_packed_ue8m0']

    in_with_sf_factor = in_dtype in ('e4m3', 'e2m1')
    source_dtype = 'bf16' if in_with_sf_factor else in_dtype
    x = generate_rand_float((num_tokens, hidden)).to(str_to_dtype(source_dtype))
    original_x = x

    if in_with_sf_factor:
        x_data, x_sf = tile_kernels.torch.cast(
            x,
            in_dtype,
            (1, num_per_channels),
            use_tma_aligned_col_major_sf=in_use_tma_aligned_col_major_sf,
            round_sf=in_round_sf,
            use_packed_ue8m0=in_use_packed_ue8m0,
        )
        sf_strides = list(x_sf.stride())
        stride_dim = 1 if in_use_tma_aligned_col_major_sf else 0
        sf_strides[stride_dim] = random.randint(sf_strides[stride_dim], 2 * sf_strides[stride_dim])
        x_sf_strided = torch.empty_strided(x_sf.shape, sf_strides, dtype=x_sf.dtype, device=x_sf.device)
        x_sf_strided.copy_(x_sf)
        x = (x_data, x_sf_strided)

    return x, original_x


def generate_test_params(level: int) -> list[dict]:
    hidden_sizes = generate_hidden_sizes(
        level, ['main', 'ffn', '2*ffn', 'q_lora', '128*indexer', 'wv', 'head', 'attn_context', 'engram_input', '5*main'], align=128
    )
    return [
        {
            'num_per_tokens': num_per_tokens,
            'num_tokens': num_tokens,
            'hidden': hidden_size,
            'round_sf': out_round_sf,
            'in_dtype': in_dtype,
            'num_per_channels': num_per_channels,
            'use_packed_ue8m0': out_use_packed_ue8m0,
            'in_use_tma_aligned_col_major_sf': in_use_tma_aligned_col_major_sf,
            'in_round_sf': in_round_sf,
            'in_use_packed_ue8m0': in_use_packed_ue8m0,
        }
        for num_per_tokens in (32,)
        for num_tokens in generate_num_tokens(level, 128, backward_only=True)
        for hidden_size in hidden_sizes
        for in_dtype in ('bf16', 'fp32', 'e4m3')
        for in_use_tma_aligned_col_major_sf, in_round_sf, in_use_packed_ue8m0 in generate_cast_config(level, in_dtype)
        for out_round_sf, out_use_packed_ue8m0 in (((False, False), (True, True)) if level >= 1 else ((True, True),))
        for num_per_channels in ((32,) if in_dtype == 'e4m3' else (None,))
    ]


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_per_channel_cast(params):
    num_tokens = params['num_tokens']
    num_per_tokens = params['num_per_tokens']
    num_per_channels = params['num_per_channels']
    round_sf = params['round_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']

    x, original_x = generate_test_data(params)
    x_fp8, per_channel_sf_inv = tile_kernels.quant.per_channel_cast(x, 'e4m3', num_per_tokens, round_sf, num_per_channels, use_packed_ue8m0)
    x_fp8_ref, per_channel_sf_inv_ref = tile_kernels.torch.cast(
        x,
        'e4m3',
        block_size=(num_per_tokens, 1),
        x_block_size=(1, num_per_channels) if num_per_channels is not None else None,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )

    assert_equal(x_fp8, x_fp8_ref)

    # Check bias
    x_casted_back = tile_kernels.torch.cast_back((x_fp8_ref, per_channel_sf_inv_ref), 'fp32', (num_per_tokens, 1))
    check_bias(x_casted_back, original_x)

    if use_packed_ue8m0:
        per_channel_sf_inv = clear_unused_sf_col_pack(per_channel_sf_inv, num_tokens, num_per_tokens)
        per_channel_sf_inv_ref = clear_unused_sf_col_pack(per_channel_sf_inv_ref, num_tokens, num_per_tokens)
    assert_equal(per_channel_sf_inv, per_channel_sf_inv_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0), ids=make_param_id)
def test_per_channel_cast_benchmark(benchmark_timer, benchmark_record, params):
    num_per_tokens = params['num_per_tokens']
    num_per_channels = params['num_per_channels']
    round_sf = params['round_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']

    x, _ = generate_test_data(params)
    x_fp8, per_channel_sf_inv = tile_kernels.quant.per_channel_cast(x, 'e4m3', num_per_tokens, round_sf, num_per_channels, use_packed_ue8m0)

    t_us = benchmark_timer(lambda: tile_kernels.quant.per_channel_cast(x, 'e4m3', num_per_tokens, round_sf, num_per_channels, use_packed_ue8m0))
    num_bytes = count_bytes(x, x_fp8, per_channel_sf_inv)

    benchmark_record(
        kernel='per_channel_cast',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )

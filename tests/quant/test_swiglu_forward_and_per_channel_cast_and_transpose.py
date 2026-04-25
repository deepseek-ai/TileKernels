import os
import pytest
import torch

import tile_kernels
from tile_kernels.testing.generator import generate_hidden_sizes, generate_num_tokens
from tile_kernels.testing.numeric import assert_equal, count_bytes
from tile_kernels.testing.bench import make_param_id
from tests.conftest import IS_HIP

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'

# HIP compilation fix: tilelang codegen_hip.cc now handles ShuffleNode for
# bfloat16x2/float16x2 using __pack_bfloat162/__pack_half2 with ROCm's uint1.


def generate_test_data(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    dtype = torch.bfloat16
    x = torch.randn((num_tokens, hidden * 2), dtype=dtype, device='cuda')
    return x


def generate_test_params(is_benchmark: bool) -> list[dict]:
    params = [
        {
            'num_tokens': num_tokens,
            'hidden': hidden_size,
            'num_per_tokens': num_per_tokens,
            'without_transpose': without_transpose,
            'round_sf': round_sf,
            'swiglu_clamp_value': swiglu_clamp_value,
        }
        for num_tokens in generate_num_tokens(128, is_benchmark=is_benchmark)
        for hidden_size in generate_hidden_sizes()
        for num_per_tokens in (32, 128)
        for without_transpose in (True, False)
        for round_sf in (True, False)
        for swiglu_clamp_value in (None, 10.0, 0.5)
    ]
    if is_benchmark:
        params = [p for p in params if p['swiglu_clamp_value'] in (None, 0.5)]
    return params


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_swiglu_forward_and_per_channel_cast_and_transpose(params):
    num_per_tokens = params['num_per_tokens']
    without_transpose = params['without_transpose']
    round_sf = params['round_sf']
    swiglu_clamp_value = params['swiglu_clamp_value']

    x = generate_test_data(params)

    func = lambda: tile_kernels.quant.swiglu_forward_and_per_channel_cast_and_transpose(
        x,
        'e4m3',
        num_per_tokens=num_per_tokens,
        round_sf=round_sf,
        without_transpose=without_transpose,
        swiglu_clamp_value=swiglu_clamp_value,
    )

    def func_ref():
        act_out = tile_kernels.torch.swiglu_forward(
            x,
            swiglu_clamp_value=swiglu_clamp_value,
        ).bfloat16()
        x_fp8_ref, x_sf_ref = tile_kernels.torch.cast(
            act_out,
            'e4m3',
            block_size=(num_per_tokens, 1),
            round_sf=round_sf,
        )
        if not without_transpose:
            x_fp8_ref = x_fp8_ref.T.contiguous()
        return x_fp8_ref, x_sf_ref

    x_fp8, x_sf = func()
    x_fp8_ref, x_sf_ref = func_ref()

    assert_equal(x_sf, x_sf_ref)
    assert_equal(x_fp8, x_fp8_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_swiglu_forward_and_per_channel_cast_and_transpose_benchmark(benchmark_timer, benchmark_record, params):
    num_per_tokens = params['num_per_tokens']
    without_transpose = params['without_transpose']
    round_sf = params['round_sf']
    swiglu_clamp_value = params['swiglu_clamp_value']

    x = generate_test_data(params)

    func = lambda: tile_kernels.quant.swiglu_forward_and_per_channel_cast_and_transpose(
        x,
        'e4m3',
        num_per_tokens=num_per_tokens,
        round_sf=round_sf,
        without_transpose=without_transpose,
        swiglu_clamp_value=swiglu_clamp_value,
    )

    x_fp8, x_sf = func()
    num_bytes = count_bytes(x, x_fp8, x_sf)

    t_us = benchmark_timer(func)

    benchmark_record(
        kernel='swiglu_forward_and_per_channel_cast_and_transpose',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )

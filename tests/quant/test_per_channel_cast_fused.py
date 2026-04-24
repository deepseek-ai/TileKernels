import itertools
import os
import pytest
import torch

import tile_kernels
from tile_kernels.testing.generator import generate_topk_idx, generate_hidden_sizes, generate_moe_params
from tile_kernels.testing.numeric import assert_equal, count_bytes
from tile_kernels.testing.bench import make_param_id
from tile_kernels.torch.per_channel_cast_fused import per_channel_cast_fused as torch_ref_per_channel_cast_fused
from tests.conftest import IS_HIP

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'

# per_channel_cast_fused depends on get_fused_mapping which uses T.sync_warp() not supported on HIP/AMD targets
pytestmark = pytest.mark.skipif(IS_HIP, reason='per_channel_cast_fused depends on get_fused_mapping (T.sync_warp()) not supported on HIP/AMD')


def generate_test_data(params):
    num_send_tokens = params['num_send_tokens']
    num_topk = params['num_topk']
    num_experts = params['num_experts']
    hidden = params['hidden']
    num_per_tokens = params['num_per_tokens']
    num_per_channels = params['num_per_channels']
    is_fused_cast_back = params['is_fused_cast_back']
    round_sf = params['round_sf']

    pos_to_token = None
    if num_topk > 0:
        topk_idx = generate_topk_idx(params)
        num_tokens = topk_idx.shape[0]
        _, pos_to_token, _, token_topk_to_pos, _, _, _, _ = (
            tile_kernels.moe.get_fused_mapping(topk_idx, num_experts, 0, 128)
        )
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        x = tile_kernels.moe.expand_to_fused(x, token_topk_to_pos, pos_to_token)
    else:
        num_tokens = num_send_tokens
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')

    if is_fused_cast_back:
        x = tile_kernels.quant.per_token_cast(x, 'e4m3', num_per_channels)

    func = lambda: tile_kernels.quant.per_channel_cast_fused(
        x, 'e4m3', num_per_tokens=num_per_tokens, round_sf=round_sf,
        num_per_channels=num_per_channels if is_fused_cast_back else None,
        pos_to_token=pos_to_token,
    )
    func_ref = lambda: torch_ref_per_channel_cast_fused(x, num_per_tokens, num_per_channels, round_sf, pos_to_token)

    return (x, num_tokens, pos_to_token, func, func_ref)


def generate_test_params(is_benchmark: bool) -> list[dict]:
    params = [
        {
            **moe,
            'hidden': hidden_size,
            'num_per_tokens': num_per_tokens,
            'num_per_channels': num_per_channels,
            'is_fused_cast_back': is_fused_cast_back,
            'round_sf': round_sf,
        }
        for moe in itertools.chain(
            iter([{'num_send_tokens': 4096, 'num_topk': 0, 'num_experts': 0, 'num_ep_ranks': 0}]),
            generate_moe_params(),
        )
        for hidden_size in generate_hidden_sizes(128)
        for num_per_tokens, num_per_channels in [(128, 128)]
        for is_fused_cast_back in (False, True)
        for round_sf in (False, True)
    ]
    if is_benchmark:
        params = [p for p in params if p['num_topk'] in (0, 6)]
    return params


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_per_channel_cast_fused(params):
    _, _, _, func, func_ref = generate_test_data(params)

    x_fp8, x_fp8_sf = func()
    x_fp8_ref, x_fp8_sf_ref = func_ref()

    assert_equal(x_fp8, x_fp8_ref)
    assert_equal(x_fp8_sf, x_fp8_sf_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_per_channel_cast_fused_benchmark(benchmark_timer, benchmark_record, params):
    x, num_tokens, pos_to_token, func, func_ref = generate_test_data(params)

    x_fp8, x_fp8_sf = func()

    t_us = benchmark_timer(func)
    num_bytes = count_bytes(x, pos_to_token, x_fp8, x_fp8_sf)

    params.pop('num_send_tokens')
    benchmark_record(
        kernel='per_channel_cast_fused',
        operation='fwd',
        params={'num_tokens': num_tokens, **params},
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )

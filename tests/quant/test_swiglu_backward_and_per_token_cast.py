import os
import pytest
import torch

import tile_kernels
from tile_kernels.testing.generator import generate_topk_idx, generate_hidden_sizes, generate_moe_params
from tile_kernels.testing.numeric import assert_equal, calc_diff, count_bytes
from tile_kernels.testing.bench import make_param_id
from tests.conftest import IS_HIP

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'

# swiglu_backward_and_per_token_cast depends on get_fused_mapping which uses T.sync_warp() not supported on HIP/AMD targets
pytestmark = pytest.mark.skipif(IS_HIP, reason='swiglu_backward depends on get_fused_mapping (T.sync_warp()) not supported on HIP/AMD')


def generate_test_data(params):
    num_topk = params['num_topk']
    num_experts = params['num_experts']
    hidden = params['hidden']
    num_per_channels = params['num_per_channels']

    topk_idx = generate_topk_idx(params)
    num_tokens = topk_idx.shape[0]
    topk_weights = torch.rand((num_tokens, num_topk), dtype=torch.float32, device='cuda')
    x = torch.randn((num_tokens, hidden * 2), dtype=torch.bfloat16, device='cuda')
    _, pos_to_token, pos_to_token_topk, token_topk_to_pos, _, _, _, _ = tile_kernels.moe.get_fused_mapping(
        topk_idx, num_experts, 0, num_per_channels
    )
    x_expand = tile_kernels.moe.expand_to_fused(x, token_topk_to_pos, pos_to_token)
    x = tile_kernels.quant.per_token_cast(x_expand, 'e4m3', num_per_channels=num_per_channels)
    num_expand_tokens = x_expand.size(0)
    weighted_act_x_grad = torch.randn((num_expand_tokens, hidden), dtype=torch.bfloat16, device='cuda')

    return (x, num_tokens, topk_weights, pos_to_token_topk, token_topk_to_pos, weighted_act_x_grad)


def generate_test_params(is_benchmark: bool) -> list[dict]:
    params = [
        {
            'num_send_tokens': moe['num_send_tokens'],
            'num_topk': moe['num_topk'],
            'num_experts': moe['num_experts'],
            'num_ep_ranks': moe['num_ep_ranks'],
            'hidden': hidden_size,
            'num_per_channels': num_per_channels,
            'round_sf': round_sf,
            'swiglu_clamp_value': swiglu_clamp_value,
        }
        for moe in generate_moe_params(is_benchmark)
        for hidden_size in generate_hidden_sizes(256)
        for num_per_channels in (32, 128)
        for round_sf in (False, True)
        for swiglu_clamp_value in (None, 10.0, 0.5)
    ]
    if is_benchmark:
        params = [p for p in params if p['swiglu_clamp_value'] in (None, 0.5) and p['round_sf']]
    return params


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_swiglu_backward_and_per_token_cast(params):
    num_per_channels = params['num_per_channels']
    round_sf = params['round_sf']
    swiglu_clamp_value = params['swiglu_clamp_value']

    x, num_tokens, topk_weights, pos_to_token_topk, token_topk_to_pos, weighted_act_x_grad = generate_test_data(params)

    func = lambda: tile_kernels.quant.swiglu_backward_and_per_token_cast(
        x,
        weighted_act_x_grad,
        topk_weights,
        pos_to_token_topk,
        token_topk_to_pos,
        num_per_channels=num_per_channels,
        round_sf=round_sf,
        swiglu_clamp_value=swiglu_clamp_value,
    )

    def func_ref():
        out, x_grad_full, weight_grad = tile_kernels.torch.swiglu_backward(
            x,
            weighted_act_x_grad,
            topk_weights,
            pos_to_token_topk,
            token_topk_to_pos,
            num_per_channels=num_per_channels,
            swiglu_clamp_value=swiglu_clamp_value,
        )
        x_grad_fp8_full, x_grad_fp8_sf = tile_kernels.torch.cast(
            x_grad_full, 'e4m3', block_size=(1, num_per_channels), round_sf=round_sf
        )
        return (
            out.to(weighted_act_x_grad.dtype),
            (x_grad_fp8_full, x_grad_fp8_sf),
            x_grad_full.to(weighted_act_x_grad.dtype),
            weight_grad,
        )

    weighted_act_x, x_grad_fp8, x_grad, topk_weights_grad = func()
    weighted_act_x_ref, x_grad_fp8_ref, x_grad_ref, topk_weights_grad_ref = func_ref()

    assert_equal(weighted_act_x, weighted_act_x_ref)
    assert_equal(x_grad, x_grad_ref)
    assert_equal(x_grad_fp8[0], x_grad_fp8_ref[0])
    assert_equal(x_grad_fp8[1], x_grad_fp8_ref[1])
    torch.testing.assert_close(topk_weights_grad, topk_weights_grad_ref, atol=1e-4, rtol=1e-4)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_swiglu_backward_and_per_token_cast_benchmark(benchmark_timer, benchmark_record, params):
    num_per_channels = params['num_per_channels']
    round_sf = params['round_sf']
    swiglu_clamp_value = params['swiglu_clamp_value']

    x, num_tokens, topk_weights, pos_to_token_topk, token_topk_to_pos, weighted_act_x_grad = generate_test_data(params)

    func = lambda: tile_kernels.quant.swiglu_backward_and_per_token_cast(
        x,
        weighted_act_x_grad,
        topk_weights,
        pos_to_token_topk,
        token_topk_to_pos,
        num_per_channels=num_per_channels,
        round_sf=round_sf,
        swiglu_clamp_value=swiglu_clamp_value,
    )

    weighted_act_x, x_grad_fp8, x_grad, topk_weights_grad = func()

    t_us = benchmark_timer(func)
    num_bytes = count_bytes(
        x,
        weighted_act_x_grad,
        topk_weights,
        pos_to_token_topk,
        token_topk_to_pos,
        weighted_act_x,
        x_grad_fp8,
        x_grad,
        topk_weights_grad,
    )

    params.pop('num_send_tokens')
    benchmark_record(
        kernel='swiglu_backward_and_per_token_cast',
        operation='fwd',
        params={'num_tokens': num_tokens, **params},
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )

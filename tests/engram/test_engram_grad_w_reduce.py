import os

import pytest
import torch

from tile_kernels.config import get_device, is_ascend, set_num_sms
from tile_kernels.engram import grad_w_reduce
from tile_kernels.rand import randn
from tile_kernels.testing.bench import make_param_id
from tile_kernels.testing.generator import generate_hidden_sizes, generate_num_sms, get_test_level
from tile_kernels.testing.numeric import count_bytes
from tile_kernels.utils import str_to_dtype

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def grad_w_reduce_ref(grad_w_partial, weight_hidden, weight_embed, grad_weight_hidden, grad_weight_embed, grad_weight_value=None):
    hc_mult = weight_hidden.shape[0]
    grad_w_sum = grad_w_partial.sum(0)
    grad_weight_hidden += grad_w_sum[:hc_mult] * weight_embed.float()
    grad_weight_embed += grad_w_sum[:hc_mult] * weight_hidden.float()
    if grad_weight_value is not None:
        grad_weight_value += grad_w_sum[hc_mult]


def generate_test_data(params):
    hidden_size = params['hidden']
    dtype = str_to_dtype(params['dtype'])
    hc_mult = 4
    num_persistent_blocks = params['num_sms'] * (2 if is_ascend() else 1)
    device = get_device()
    grad_w_partial = randn(num_persistent_blocks, hc_mult + params['has_value_norm'], hidden_size, dtype=torch.float32, device=device)
    weight_hidden = randn(hc_mult, hidden_size, dtype=dtype, device=device)
    weight_embed = randn(hc_mult, hidden_size, dtype=dtype, device=device)
    return (grad_w_partial, weight_hidden, weight_embed)


def generate_test_params(level: int) -> list[dict]:
    return [
        {
            'hidden': hidden_size,
            'num_sms': num_sms,
            'dtype': dtype,
            'has_value_norm': has_value_norm,
        }
        for hidden_size in generate_hidden_sizes(level, ['main'], align=128)
        for num_sms in generate_num_sms(level)
        for dtype in ['bf16', 'fp32']
        for has_value_norm in [False, True]
    ]


@pytest.mark.parametrize('params', generate_test_params(max(get_test_level(), 1)), ids=make_param_id)
def test_engram_grad_w_reduce(params):
    hidden_size = params['hidden']

    if not is_ascend():
        set_num_sms(params['num_sms'])
    grad_w_partial, weight_hidden, weight_embed = generate_test_data(params)
    hc_mult = weight_hidden.shape[0]

    # Correctness
    device = get_device()
    grad_wh_ref = randn(hc_mult, hidden_size, dtype=torch.float32, device=device)
    grad_we_ref = randn(hc_mult, hidden_size, dtype=torch.float32, device=device)
    grad_wv_ref = randn(hidden_size, dtype=torch.float32, device=device) if params['has_value_norm'] else None
    grad_weight_hidden = grad_wh_ref.clone()
    grad_weight_embed = grad_we_ref.clone()
    grad_weight_value = grad_wv_ref.clone() if grad_wv_ref is not None else None
    grad_w_reduce_ref(grad_w_partial, weight_hidden, weight_embed, grad_wh_ref, grad_we_ref, grad_wv_ref)
    grad_w_reduce(grad_w_partial, weight_hidden, weight_embed, grad_weight_hidden, grad_weight_embed, grad_weight_value)
    torch.testing.assert_close(grad_weight_hidden, grad_wh_ref, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(grad_weight_embed, grad_we_ref, rtol=2e-5, atol=2e-5)
    if grad_weight_value is not None:
        torch.testing.assert_close(grad_weight_value, grad_wv_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(1), ids=make_param_id)
def test_engram_grad_w_reduce_benchmark(benchmark_timer, benchmark_record, params):
    hidden_size = params['hidden']

    if not is_ascend():
        set_num_sms(params['num_sms'])
    grad_w_partial, weight_hidden, weight_embed = generate_test_data(params)
    hc_mult = weight_hidden.shape[0]
    device = get_device()
    grad_weight_hidden = randn(hc_mult, hidden_size, dtype=torch.float32, device=device)
    grad_weight_embed = randn(hc_mult, hidden_size, dtype=torch.float32, device=device)
    grad_weight_value = randn(hidden_size, dtype=torch.float32, device=device) if params['has_value_norm'] else None

    t_us = benchmark_timer(
        lambda: grad_w_reduce(grad_w_partial, weight_hidden, weight_embed, grad_weight_hidden, grad_weight_embed, grad_weight_value)
    )

    num_bytes = count_bytes(grad_w_partial, weight_hidden, weight_embed, grad_weight_hidden, grad_weight_embed, grad_weight_value)
    bandwidth_gbs = num_bytes / t_us / 1e3
    benchmark_record(
        kernel='grad_w_reduce',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )

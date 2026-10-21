import os

import pytest
import torch

from tile_kernels.config import get_device
from tile_kernels.engram import engram_gate_bwd
from tile_kernels.rand import randn
from tile_kernels.testing.bench import make_param_id
from tile_kernels.testing.generator import generate_hidden_sizes, generate_num_tokens, get_test_level
from tile_kernels.testing.numeric import assert_equal, count_bytes
from tile_kernels.torch.engram import engram_gate_ref

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_test_data(params):
    num_tokens = params['num_tokens']
    hc_mult = params['hc']
    hidden_size = params['hidden']
    eps = 1e-20
    clamp_value = 1e-6
    device = get_device()
    x_data = randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    kv_data = randn(num_tokens, hc_mult + 1, hidden_size, dtype=torch.bfloat16, device=device)
    wh_data = randn(hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    we_data = randn(hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    weight_fused = wh_data.float() * we_data.float()
    grad_out = randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    image_token_mask = torch.arange(num_tokens, device=device) % 5 < 2 if params['with_mask'] else None
    return (x_data, kv_data, wh_data, we_data, weight_fused, grad_out, eps, clamp_value, image_token_mask)


def generate_test_params(level: int, mask_options: tuple[bool, ...] = (False, True)) -> list[dict]:
    return [
        {'num_tokens': t, 'hc': hc, 'hidden': hidden_size, 'with_mask': with_mask}
        for t in generate_num_tokens(level)
        for hc in (4,)
        for hidden_size in generate_hidden_sizes(256)
        for with_mask in mask_options
    ]


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_engram_gate_bwd(params):
    x_data, kv_data, wh_data, we_data, weight_fused, grad_out, eps, clamp_value, image_token_mask = generate_test_data(params)
    # Reference: forward with intermediates + autograd backward
    x_ref = x_data.clone().requires_grad_(True)
    kv_ref = kv_data.clone().requires_grad_(True)
    # Cast to float32 so autograd produces fp32 gradients matching the kernel
    wh_ref = wh_data.float().requires_grad_(True)
    we_ref = we_data.float().requires_grad_(True)
    o_ref, dot_ref, gate_score_ref, rstd_x_ref, rstd_k_ref = engram_gate_ref(
        x_ref,
        kv_ref,
        wh_ref,
        we_ref,
        clamp_value,
        eps,
        save_for_backward=True,
        image_token_mask=image_token_mask,
    )
    o_ref.backward(grad_out)

    # Kernel backward using ref intermediates
    grad_x, grad_kv, grad_w_partial = engram_gate_bwd(
        grad_out,
        x_data,
        kv_data,
        weight_fused,
        dot_ref,
        gate_score_ref,
        rstd_x_ref,
        rstd_k_ref,
        clamp_value,
        image_token_mask=image_token_mask,
    )
    grad_w_fused = grad_w_partial.sum(0)
    grad_wh = grad_w_fused * we_data.float()
    grad_we = grad_w_fused * wh_data.float()

    # Correctness
    torch.testing.assert_close(grad_x, x_ref.grad, rtol=8e-3, atol=8e-3)
    torch.testing.assert_close(grad_kv, kv_ref.grad, rtol=7e-3, atol=7e-3)
    torch.testing.assert_close(grad_wh, wh_ref.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(grad_we, we_ref.grad, rtol=2e-4, atol=2e-4)
    if image_token_mask is not None:
        assert_equal(grad_x[image_token_mask], grad_out[image_token_mask])
        assert_equal(grad_kv[image_token_mask], torch.zeros_like(grad_kv[image_token_mask]))


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0, mask_options=(False,)), ids=make_param_id)
def test_engram_gate_bwd_benchmark(benchmark_timer, benchmark_record, params):
    (x_data, kv_data, wh_data, we_data, weight_fused, grad_out, eps, clamp_value, _) = generate_test_data(params)

    # Forward to get intermediates
    o_ref, dot_ref, gate_score_ref, rstd_x_ref, rstd_k_ref = engram_gate_ref(
        x_data,
        kv_data,
        wh_data,
        we_data,
        clamp_value,
        eps,
        save_for_backward=True,
    )

    grad_x, grad_kv, grad_w_partial = engram_gate_bwd(
        grad_out,
        x_data,
        kv_data,
        weight_fused,
        dot_ref,
        gate_score_ref,
        rstd_x_ref,
        rstd_k_ref,
        clamp_value,
    )

    func_bwd = lambda: engram_gate_bwd(
        grad_out,
        x_data,
        kv_data,
        weight_fused,
        dot_ref,
        gate_score_ref,
        rstd_x_ref,
        rstd_k_ref,
        clamp_value,
    )
    t_us = benchmark_timer(func_bwd)
    num_bytes = count_bytes(
        grad_out,
        x_data,
        kv_data,
        weight_fused,
        dot_ref,
        gate_score_ref,
        rstd_x_ref,
        rstd_k_ref,
        grad_x,
        grad_kv,
        grad_w_partial,
    )
    benchmark_record(
        kernel='engram_gate_bwd',
        operation='bwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )

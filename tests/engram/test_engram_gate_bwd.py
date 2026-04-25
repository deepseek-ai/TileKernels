import os
import pytest
import torch

from tile_kernels.engram import engram_gate_bwd
from tile_kernels.torch.engram import engram_gate_ref
from tile_kernels.testing.numeric import calc_diff, count_bytes
from tile_kernels.testing.generator import generate_hidden_sizes, generate_num_tokens
from tile_kernels.testing.bench import make_param_id
from tests.conftest import IS_HIP

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_test_data(params):
    num_tokens = params['num_tokens']
    hc_mult = params['hc']
    hidden_size = params['hidden']
    eps = 1e-20
    clamp_value = 1e-6
    x_data = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device='cuda')
    k_data = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device='cuda')
    v_data = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device='cuda')
    wh_data = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device='cuda')
    we_data = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device='cuda')
    weight_fused = wh_data.float() * we_data.float()
    grad_out = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device='cuda')
    return (x_data, k_data, v_data, wh_data, we_data, weight_fused, grad_out, eps, clamp_value)


def generate_test_params(is_benchmark: bool) -> list[dict]:
    return [
        {'num_tokens': t, 'hc': hc, 'hidden': hidden_size}
        for t in generate_num_tokens(is_benchmark=is_benchmark)
        for hc in (4,)
        for hidden_size in generate_hidden_sizes(128)
    ]


@pytest.mark.skipif(IS_HIP, reason='engram_gate_bwd uses T.get_warp_idx() which is not supported on HIP/AMD targets')
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_engram_gate_bwd(params):
    (x_data, k_data, v_data, wh_data, we_data, weight_fused, grad_out, eps, clamp_value) = generate_test_data(params)

    # Reference: forward with intermediates + autograd backward
    x_ref = x_data.clone().requires_grad_(True)
    k_ref = k_data.clone().requires_grad_(True)
    v_ref = v_data.clone().requires_grad_(True)
    # Cast to float32 so autograd produces fp32 gradients matching the kernel
    wh_ref = wh_data.float().requires_grad_(True)
    we_ref = we_data.float().requires_grad_(True)
    o_ref, dot_ref, gate_score_ref, rstd_x_ref, rstd_k_ref = engram_gate_ref(
        x_ref, k_ref, v_ref, wh_ref, we_ref, clamp_value, eps, save_for_backward=True,
    )
    o_ref.backward(grad_out)

    # Kernel backward using ref intermediates
    grad_x, grad_k, grad_v, grad_w_partial = engram_gate_bwd(
        grad_out, x_data, k_data, v_data, weight_fused,
        dot_ref, gate_score_ref, rstd_x_ref, rstd_k_ref, clamp_value,
    )
    grad_w_fused = grad_w_partial.sum(0)
    grad_wh = grad_w_fused * we_data.float()
    grad_we = grad_w_fused * wh_data.float()

    # Correctness
    diff_x = calc_diff(grad_x, x_ref.grad)
    assert diff_x < 1e-8, f'grad_x mismatch: {diff_x:.6e}'
    diff_k = calc_diff(grad_k, k_ref.grad)
    assert diff_k < 1e-8, f'grad_k mismatch: {diff_k:.6e}'
    diff_v = calc_diff(grad_v, v_ref.grad)
    assert diff_v < 1e-8, f'grad_v mismatch: {diff_v:.6e}'
    diff_wh = calc_diff(grad_wh, wh_ref.grad)
    assert diff_wh < 1e-8, f'grad_wh mismatch: {diff_wh:.6e}'
    diff_we = calc_diff(grad_we, we_ref.grad)
    assert diff_we < 1e-8, f'grad_we mismatch: {diff_we:.6e}'


@pytest.mark.benchmark
@pytest.mark.skipif(IS_HIP, reason='engram_gate_bwd uses T.get_warp_idx() which is not supported on HIP/AMD targets')
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_engram_gate_bwd_benchmark(benchmark_timer, benchmark_record, params):
    (x_data, k_data, v_data, wh_data, we_data, weight_fused, grad_out, eps, clamp_value) = generate_test_data(params)

    # Forward to get intermediates
    o_ref, dot_ref, gate_score_ref, rstd_x_ref, rstd_k_ref = engram_gate_ref(
        x_data, k_data, v_data, wh_data, we_data, clamp_value, eps, save_for_backward=True,
    )

    grad_x, grad_k, grad_v, grad_w_partial = engram_gate_bwd(
        grad_out, x_data, k_data, v_data, weight_fused,
        dot_ref, gate_score_ref, rstd_x_ref, rstd_k_ref, clamp_value,
    )
    grad_w_fused = grad_w_partial.sum(0)
    grad_wh = grad_w_fused * we_data.float()
    grad_we = grad_w_fused * wh_data.float()

    func_bwd = lambda: engram_gate_bwd(
        grad_out, x_data, k_data, v_data, weight_fused,
        dot_ref, gate_score_ref, rstd_x_ref, rstd_k_ref, clamp_value,
    )
    t_us = benchmark_timer(func_bwd)
    num_bytes = count_bytes(
        grad_out, x_data, k_data, v_data, weight_fused,
        dot_ref, gate_score_ref, rstd_x_ref, rstd_k_ref,
        grad_x, grad_k, grad_v, grad_wh, grad_we,
    )
    benchmark_record(
        kernel='engram_gate_bwd',
        operation='bwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )

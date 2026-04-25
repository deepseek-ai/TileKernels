import os
import pytest
import torch

from tile_kernels.engram import engram_gate_fwd
from tile_kernels.torch.engram import engram_gate_ref
from tile_kernels.testing.numeric import assert_equal, calc_diff, count_bytes
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
    return (x_data, k_data, v_data, wh_data, we_data, weight_fused, eps, clamp_value)


def generate_test_params(is_benchmark: bool) -> list[dict]:
    return [
        {'num_tokens': t, 'hc': hc, 'hidden': hidden_size}
        for t in generate_num_tokens(is_benchmark=is_benchmark)
        for hc in (4,)
        for hidden_size in generate_hidden_sizes(128)
    ]


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_engram_gate_fwd(params):
    (x_data, k_data, v_data, wh_data, we_data, weight_fused, eps, clamp_value) = generate_test_data(params)

    out_ref, dot_ref, gate_score_ref, rstd_x_ref, rstd_k_ref = engram_gate_ref(
        x_data, k_data, v_data, wh_data, we_data, clamp_value, eps, save_for_backward=True,
    )

    # Correctness: save_for_backward=True
    out_save, dot, gate_score, rstd_x, rstd_k = engram_gate_fwd(
        x_data, k_data, v_data, weight_fused, eps, clamp_value, save_for_backward=True,
    )
    # HIP (hipcc/clang) may use different FMA contraction patterns than CUDA
    # (nvcc/ptx) for the bfloat16 output computation (x + gate_score * v),
    # producing 1-2 ULP differences that marginally exceed the CUDA threshold.
    # Relax the output threshold slightly for HIP while keeping all other
    # checks at the original 2e-10.
    out_threshold = 5e-10 if IS_HIP else 2e-10

    assert dot is not None and gate_score is not None and rstd_x is not None and rstd_k is not None
    diff_out = calc_diff(out_save, out_ref)
    assert diff_out < out_threshold, f'out_save mismatch: {diff_out:.6e}'
    diff_dot = calc_diff(dot, dot_ref)
    assert diff_dot < 2e-10, f'dot mismatch: {diff_dot:.6e}'
    diff_gate = calc_diff(gate_score, gate_score_ref)
    assert diff_gate < 2e-10, f'gate_score mismatch: {diff_gate:.6e}'
    diff_rstd_x = calc_diff(rstd_x, rstd_x_ref)
    assert diff_rstd_x < 2e-10, f'rstd_x mismatch: {diff_rstd_x:.6e}'
    diff_rstd_k = calc_diff(rstd_k, rstd_k_ref)
    assert diff_rstd_k < 2e-10, f'rstd_k mismatch: {diff_rstd_k:.6e}'

    # Correctness: save_for_backward=False
    out_no_save, dot_n, gate_score_n, rstd_x_n, rstd_k_n = engram_gate_fwd(
        x_data, k_data, v_data, weight_fused, eps, clamp_value, save_for_backward=False,
    )
    assert dot_n is None and gate_score_n is None and rstd_x_n is None and rstd_k_n is None
    diff_out = calc_diff(out_no_save, out_ref)
    assert diff_out < out_threshold, f'out_no_save mismatch: {diff_out:.6e}'
    assert_equal(out_no_save, out_save)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_engram_gate_fwd_benchmark(benchmark_timer, benchmark_record, params):
    (x_data, k_data, v_data, _, _, weight_fused, eps, clamp_value) = generate_test_data(params)

    # Benchmark save_for_backward=True
    out_save, dot, gate_score, rstd_x, rstd_k = engram_gate_fwd(
        x_data, k_data, v_data, weight_fused, eps, clamp_value, save_for_backward=True,
    )
    t_save_us = benchmark_timer(lambda: engram_gate_fwd(
        x_data, k_data, v_data, weight_fused, eps, clamp_value, save_for_backward=True,
    ))
    num_bytes_save = count_bytes(x_data, k_data, v_data, weight_fused, out_save, dot, gate_score, rstd_x, rstd_k)
    benchmark_record(
        kernel='engram_gate_fwd',
        operation='fwd',
        params={**params, 'save': True},
        time_us=t_save_us,
        bandwidth_gbs=num_bytes_save / t_save_us / 1e3,
    )

    # Benchmark save_for_backward=False
    out_no_save = engram_gate_fwd(
        x_data, k_data, v_data, weight_fused, eps, clamp_value, save_for_backward=False,
    )[0]
    t_no_save_us = benchmark_timer(lambda: engram_gate_fwd(
        x_data, k_data, v_data, weight_fused, eps, clamp_value, save_for_backward=False,
    ))
    num_bytes_no_save = count_bytes(x_data, k_data, v_data, weight_fused, out_no_save)
    benchmark_record(
        kernel='engram_gate_fwd',
        operation='fwd',
        params={**params, 'save': False},
        time_us=t_no_save_us,
        bandwidth_gbs=num_bytes_no_save / t_no_save_us / 1e3,
    )

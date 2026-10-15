import os

import pytest
import torch

from tile_kernels.config import get_device, is_ascend
from tile_kernels.engram import (
    engram_sinkhorn_finalize,
    engram_sinkhorn_momentum_update,
    engram_sinkhorn_step,
    engram_sinkhorn_step_reduce,
)
from tile_kernels.rand import randn
from tile_kernels.testing.bench import make_param_id
from tile_kernels.testing.generator import generate_num_sms, get_test_level
from tile_kernels.testing.numeric import assert_equal, count_bytes
from tile_kernels.torch.engram import (
    engram_sinkhorn_finalize_ref,
    engram_sinkhorn_momentum_update_ref,
    engram_sinkhorn_step_ref,
)
from tile_kernels.utils import str_to_dtype

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_momentum_test_data(params):
    numel = params['numel']
    grad_dtype = str_to_dtype(params['grad_dtype'])
    momentum_dtype = str_to_dtype(params['momentum_dtype'])
    nesterov = params['nesterov']
    beta1 = params['beta1']

    device = get_device()
    grad = randn(numel, dtype=grad_dtype, device=device)
    momentum_buffer = randn(numel, dtype=momentum_dtype, device=device)
    return grad, momentum_buffer, beta1, nesterov


def generate_momentum_test_params(level: int) -> list[dict]:
    sizes = [1001 * 256, 10001 * 256, 100001 * 512, 1000001 * 512]
    grad_dtypes = ('fp32',) if level == 0 else ('bf16', 'fp32')
    return [
        {'numel': numel, 'grad_dtype': grad_dtype, 'momentum_dtype': momentum_dtype, 'nesterov': nesterov, 'beta1': beta1}
        for numel in sizes
        for grad_dtype in grad_dtypes
        for momentum_dtype in ('bf16', 'fp32')
        for nesterov in (False, True)
        for beta1 in (0.95,)
    ]


@pytest.mark.parametrize('params', generate_momentum_test_params(get_test_level()), ids=make_param_id)
def test_engram_sinkhorn_momentum_update(params):
    grad, momentum_buffer, beta1, nesterov = generate_momentum_test_data(params)
    momentum_buffer_ref = momentum_buffer.clone()

    sinkhorn_input_ref = engram_sinkhorn_momentum_update_ref(grad, momentum_buffer_ref, beta1, nesterov)
    sinkhorn_input = engram_sinkhorn_momentum_update(grad, momentum_buffer, beta1, nesterov)

    assert_equal(momentum_buffer, momentum_buffer_ref)
    assert_equal(sinkhorn_input, sinkhorn_input_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_momentum_test_params(0), ids=make_param_id)
def test_engram_sinkhorn_momentum_update_benchmark(benchmark_timer, benchmark_record, params):
    grad, momentum_buffer, beta1, nesterov = generate_momentum_test_data(params)
    sinkhorn_input = engram_sinkhorn_momentum_update(grad, momentum_buffer, beta1, nesterov)

    t_us = benchmark_timer(lambda: engram_sinkhorn_momentum_update(grad, momentum_buffer, beta1, nesterov))

    num_bytes = count_bytes(grad, momentum_buffer, momentum_buffer, sinkhorn_input)
    bandwidth_gbs = num_bytes / t_us / 1e3
    benchmark_record(
        kernel='engram_sinkhorn',
        operation='momentum_update',
        params=params,
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )


def generate_row_step_test_data(params):
    num_rows = params['num_rows']
    head_dim = params['head_dim']
    device = get_device()
    matrix = randn(num_rows, head_dim, dtype=torch.float32, device=device)
    row_scale = torch.rand(num_rows, dtype=torch.float32, device=device)
    col_scale = torch.rand(head_dim, dtype=torch.float32, device=device)
    col_sumsq = torch.zeros_like(col_scale)
    return matrix, row_scale, col_scale, col_sumsq, params['eps']


def generate_row_step_test_params(level: int) -> list[dict]:
    return [
        {'num_rows': num_rows, 'head_dim': head_dim, 'eps': eps} for num_rows in [1001, 10001, 100001] for head_dim in [256, 5120] for eps in (1e-20,)
    ]


def generate_step_reduce_test_data(params):
    device = get_device()
    col_sumsq_partial = torch.rand(params['num_partials'], params['head_dim'], dtype=torch.float32, device=device)
    col_sumsq = torch.empty(params['head_dim'], dtype=torch.float32, device=device)
    return col_sumsq_partial, col_sumsq


def generate_step_reduce_test_params(level: int) -> list[dict]:
    partials_per_sm = 2 if is_ascend() else 3
    return [{'num_partials': num_sms * partials_per_sm, 'head_dim': head_dim} for num_sms in generate_num_sms(level) for head_dim in [256, 5120]]


@pytest.mark.parametrize('params', generate_row_step_test_params(get_test_level()), ids=make_param_id)
def test_engram_sinkhorn_step(params):
    matrix, row_scale, col_scale, col_sumsq_ref, eps = generate_row_step_test_data(params)
    row_scale_ref = row_scale.clone()

    row_norm_ref = engram_sinkhorn_step_ref(matrix, row_scale_ref, col_scale, col_sumsq_ref, eps)
    row_norm, col_sumsq_partial = engram_sinkhorn_step(matrix, row_scale, col_scale, eps)
    col_sumsq = col_sumsq_partial.sum(0)

    torch.testing.assert_close(row_norm, row_norm_ref)
    torch.testing.assert_close(row_scale, row_scale_ref)
    torch.testing.assert_close(col_sumsq, col_sumsq_ref)


@pytest.mark.parametrize('params', generate_step_reduce_test_params(get_test_level()), ids=make_param_id)
def test_engram_sinkhorn_step_reduce(params):
    col_sumsq_partial, col_sumsq = generate_step_reduce_test_data(params)
    col_sumsq_ref = col_sumsq_partial.sum(0)

    engram_sinkhorn_step_reduce(col_sumsq_partial, col_sumsq)

    # The kernel accumulates in fp32 serially (row order), while torch's
    # sum uses a pairwise tree; for large num_partials the rounding
    # difference sits right at the default float32 rtol (1.3e-6) -- a pure
    # torch row-serial loop reproduces ~1.4e-6 on some columns for
    # num_partials=564. Compare against a float64 reference with an
    # explicit tolerance instead.
    torch.testing.assert_close(
        col_sumsq, col_sumsq_ref.to(torch.float64).to(torch.float32),
        rtol=3e-6, atol=1e-5)


@pytest.mark.parametrize('params', generate_row_step_test_params(get_test_level()), ids=make_param_id)
def test_engram_sinkhorn_finalize(params):
    matrix, row_scale, col_scale, _, eps = generate_row_step_test_data(params)
    alpha = params['head_dim'] ** 0.5

    out_ref = engram_sinkhorn_finalize_ref(matrix, row_scale, col_scale, eps, alpha)
    out = engram_sinkhorn_finalize(matrix, row_scale, col_scale, eps, alpha)

    torch.testing.assert_close(out, out_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_row_step_test_params(0), ids=make_param_id)
def test_engram_sinkhorn_step_benchmark(benchmark_timer, benchmark_record, params):
    matrix, row_scale, col_scale, _, eps = generate_row_step_test_data(params)

    row_norm, col_sumsq_partial = engram_sinkhorn_step(matrix, row_scale, col_scale, eps)
    t_us = benchmark_timer(lambda: engram_sinkhorn_step(matrix, row_scale, col_scale, eps))

    num_bytes = count_bytes(matrix, row_scale, col_scale, row_scale, row_norm, col_sumsq_partial)
    bandwidth_gbs = num_bytes / t_us / 1e3
    benchmark_record(
        kernel='engram_sinkhorn',
        operation='step',
        params=params,
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_step_reduce_test_params(0), ids=make_param_id)
def test_engram_sinkhorn_step_reduce_benchmark(benchmark_timer, benchmark_record, params):
    col_sumsq_partial, col_sumsq = generate_step_reduce_test_data(params)
    engram_sinkhorn_step_reduce(col_sumsq_partial, col_sumsq)

    t_us = benchmark_timer(lambda: engram_sinkhorn_step_reduce(col_sumsq_partial, col_sumsq))

    num_bytes = count_bytes(col_sumsq_partial, col_sumsq)
    bandwidth_gbs = num_bytes / t_us / 1e3
    benchmark_record(
        kernel='engram_sinkhorn',
        operation='step_reduce',
        params=params,
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_row_step_test_params(0), ids=make_param_id)
def test_engram_sinkhorn_finalize_benchmark(benchmark_timer, benchmark_record, params):
    matrix, row_scale, col_scale, _, eps = generate_row_step_test_data(params)
    alpha = params['head_dim'] ** 0.5
    out = torch.empty_like(matrix)
    engram_sinkhorn_finalize(matrix, row_scale, col_scale, eps, alpha, out)

    t_us = benchmark_timer(lambda: engram_sinkhorn_finalize(matrix, row_scale, col_scale, eps, alpha, out))

    num_bytes = count_bytes(matrix, row_scale, col_scale, out)
    bandwidth_gbs = num_bytes / t_us / 1e3
    benchmark_record(
        kernel='engram_sinkhorn',
        operation='finalize',
        params=params,
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )

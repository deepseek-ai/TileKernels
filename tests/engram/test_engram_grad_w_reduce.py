import os
import pytest
import torch

from tile_kernels.config import get_num_sms
from tile_kernels.engram import grad_w_reduce
from tile_kernels.testing.numeric import calc_diff, count_bytes
from tile_kernels.testing.generator import generate_hidden_sizes
from tile_kernels.testing.bench import make_param_id
from tests.conftest import IS_HIP

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'

# engram_grad_w_reduce_kernel fails with hipModuleLaunchKernel invalid argument for
# larger hidden sizes on HIP/AMD targets (T.Pipelined with num_stages > 1 incompatibility)
pytestmark = pytest.mark.skipif(IS_HIP, reason='engram_grad_w_reduce_kernel fails on HIP/AMD targets (invalid argument launch config)')


def grad_w_reduce_ref(grad_w_partial, weight_hidden, weight_embed, grad_weight_hidden, grad_weight_embed):
    grad_w_sum = grad_w_partial.sum(0)
    grad_weight_hidden += grad_w_sum * weight_embed.float()
    grad_weight_embed += grad_w_sum * weight_hidden.float()


def generate_test_data(params):
    hidden_size = params['hidden']
    hc_mult = 4
    num_persistent_blocks = get_num_sms()
    grad_w_partial = torch.randn(num_persistent_blocks, hc_mult, hidden_size, dtype=torch.float32, device='cuda')
    weight_hidden = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device='cuda')
    weight_embed = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device='cuda')
    return (grad_w_partial, weight_hidden, weight_embed)


def generate_test_params(is_benchmark: bool) -> list[dict]:
    return [
        {'hidden': hidden_size}
        for hidden_size in generate_hidden_sizes(128)
    ]


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_engram_grad_w_reduce(params):
    hidden_size = params['hidden']
    grad_w_partial, weight_hidden, weight_embed = generate_test_data(params)
    hc_mult = grad_w_partial.shape[1]

    # Correctness
    grad_wh_ref = torch.randn(hc_mult, hidden_size, dtype=torch.float32, device='cuda')
    grad_we_ref = torch.randn(hc_mult, hidden_size, dtype=torch.float32, device='cuda')
    grad_weight_hidden = grad_wh_ref.clone()
    grad_weight_embed = grad_we_ref.clone()
    grad_w_reduce_ref(grad_w_partial, weight_hidden, weight_embed, grad_wh_ref, grad_we_ref)
    grad_w_reduce(grad_w_partial, weight_hidden, weight_embed, grad_weight_hidden, grad_weight_embed)
    diff_wh = calc_diff(grad_weight_hidden, grad_wh_ref)
    assert diff_wh < 1e-10, f'grad_wh mismatch: {diff_wh:.6e}'
    diff_we = calc_diff(grad_weight_embed, grad_we_ref)
    assert diff_we < 1e-10, f'grad_we mismatch: {diff_we:.6e}'


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_engram_grad_w_reduce_benchmark(benchmark_timer, benchmark_record, params):
    hidden_size = params['hidden']
    grad_w_partial, weight_hidden, weight_embed = generate_test_data(params)
    hc_mult = grad_w_partial.shape[1]
    grad_weight_hidden = torch.randn(hc_mult, hidden_size, dtype=torch.float32, device='cuda')
    grad_weight_embed = torch.randn(hc_mult, hidden_size, dtype=torch.float32, device='cuda')

    t_us = benchmark_timer(lambda: grad_w_reduce(grad_w_partial, weight_hidden, weight_embed, grad_weight_hidden, grad_weight_embed))

    num_bytes = count_bytes(grad_w_partial, weight_hidden, weight_embed, grad_weight_hidden, grad_weight_embed)
    bandwidth_gbs = num_bytes / t_us / 1e3
    benchmark_record(
        kernel='grad_w_reduce',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )

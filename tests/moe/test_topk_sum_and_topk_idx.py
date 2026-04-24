import os

import torch
import pytest

import tile_kernels
from tile_kernels.testing.generator import generate_num_tokens
from tile_kernels.testing.numeric import count_bytes, assert_equal
from tile_kernels.testing.bench import make_param_id

from tile_kernels.torch import topk_sum_and_topk_group_idx as torch_topk_sum_and_topk_group_idx
from tests.conftest import IS_HIP

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'

# topk_sum_and_topk_group_idx_kernel uses T.sync_warp() which is not supported on HIP/AMD targets
pytestmark = pytest.mark.skipif(IS_HIP, reason='topk_sum_and_topk_group_idx_kernel uses T.sync_warp() not supported on HIP/AMD')


def torch_stable_topk(scores: torch.Tensor, num_topk: int):
    _, sorted_indices = torch.sort(scores, dim=1, descending=True, stable=True)
    return sorted_indices[:, :num_topk].contiguous()


def generate_test_data(params):
    num_tokens = params['num_tokens']
    num_experts = params['num_experts']
    num_groups = params['num_groups']

    num_experts_per_group = num_experts // num_groups
    scores = torch.randn((num_tokens, num_groups, num_experts_per_group), dtype=torch.float, device='cuda')

    return (scores,)


def generate_test_params(is_benchmark: bool) -> list[dict]:
    return [
        {
            'num_tokens': num_tokens,
            'num_experts': num_experts,
            'num_groups': num_groups,
            'num_group_sum_topk': num_group_sum_topk,
            'num_topk_groups': num_topk_groups,
        }
        for num_tokens in generate_num_tokens(is_benchmark=is_benchmark)
        for num_experts in (72, 256)
        for num_groups in (4, 8, 12, 16)
        if num_experts % num_groups == 0
        for num_group_sum_topk in (1, 2)
        for num_topk_groups in (2, 4)
    ]


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_topk_sum_and_topk_group_idx(params):
    (scores,) = generate_test_data(params)
    num_group_sum_topk = params['num_group_sum_topk']
    num_topk_groups = params['num_topk_groups']

    func = lambda: tile_kernels.moe.topk_sum_and_topk_group_idx(scores, num_group_sum_topk, num_topk_groups)

    group_idx_ref = torch_topk_sum_and_topk_group_idx(scores, num_group_sum_topk, num_topk_groups)
    group_idx = func()

    assert_equal(group_idx, group_idx_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_topk_sum_and_topk_group_idx_benchmark(benchmark_timer, benchmark_record, params):
    (scores,) = generate_test_data(params)
    num_group_sum_topk = params['num_group_sum_topk']
    num_topk_groups = params['num_topk_groups']

    func = lambda: tile_kernels.moe.topk_sum_and_topk_group_idx(scores, num_group_sum_topk, num_topk_groups)

    group_idx = func()

    t_us = benchmark_timer(func)
    num_bytes = count_bytes(scores, group_idx)
    bandwidth_gbs = num_bytes / t_us / 1e3

    benchmark_record(
        kernel='topk_sum_and_topk_group_idx',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )

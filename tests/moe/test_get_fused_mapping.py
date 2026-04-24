import os

import pytest

import torch

import tile_kernels
from tile_kernels.config import set_num_sms
from tile_kernels.testing.generator import generate_topk_idx, generate_moe_params, generate_num_sms
from tile_kernels.testing.numeric import count_bytes
from tile_kernels.testing.bench import make_param_id
from tests.conftest import IS_HIP

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'

# get_fused_mapping_kernel uses T.sync_warp() which is not supported on HIP/AMD targets
pytestmark = pytest.mark.skipif(IS_HIP, reason='get_fused_mapping_kernel uses T.sync_warp() not supported on HIP/AMD')


def generate_test_data(params):
    num_experts = params['num_experts']

    topk_idx = generate_topk_idx(params)
    num_tokens = topk_idx.shape[0]

    return (topk_idx, num_tokens)


def generate_test_params(is_benchmark: bool) -> list[dict]:
    params = [
        {**moe, 'alignment': alignment}
        for moe in generate_moe_params(is_benchmark=is_benchmark)
        for alignment in (64, 128)
    ]
    if is_benchmark:
        params = [{**param, 'alignment': 128} for param in params]
    return params


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_get_fused_mapping(params):
    alignment = params['alignment']

    topk_idx, num_tokens = generate_test_data(params)
    num_topk = params['num_topk']
    num_experts = params['num_experts']

    func = lambda: tile_kernels.moe.get_fused_mapping(topk_idx, num_experts, 0, alignment)

    for num_sms in generate_num_sms():
        set_num_sms(num_sms)

        pos_to_expert, pos_to_token, pos_to_token_topk, token_topk_to_pos, expert_start, expert_end, num_tokens_per_expert, num_tokens_per_expert_list = func()
        assert num_tokens_per_expert.tolist() == num_tokens_per_expert_list
        start = 0

        # Check `pos_to_expert`, `num_tokens_per_expert`, `expert_start`, `expert_end` correctness
        for i in range(num_experts):
            assert start == expert_start[i].item()
            s = pos_to_expert[start:start + num_tokens_per_expert_list[i]]
            assert (s == i).int().sum().item() == (topk_idx == i).int().sum().item()
            s = (s == i) + (s == -1)
            assert s.int().sum().item() == s.numel()
            start += num_tokens_per_expert_list[i]
            assert start == expert_end[i].item()

        non_negative_mask = pos_to_expert >= 0

        if non_negative_mask.any():
            t_values = pos_to_token_topk[non_negative_mask]
            token_indices = t_values // num_topk
            topk_indices = t_values % num_topk
            expected_indices = torch.arange(pos_to_token_topk.numel(), device='cuda')[non_negative_mask]
            actual_indices = token_topk_to_pos[token_indices, topk_indices]
            assert torch.equal(actual_indices, expected_indices)
            assert torch.equal(topk_idx[token_indices, topk_indices], pos_to_expert[non_negative_mask])
            assert torch.equal(pos_to_token_topk[non_negative_mask] // num_topk, pos_to_token[non_negative_mask])

        negative_mask = pos_to_expert < 0
        assert torch.equal(negative_mask, pos_to_token < 0)
        assert torch.equal(negative_mask, pos_to_token_topk < 0)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_get_fused_mapping_benchmark(benchmark_timer, benchmark_record, params):
    alignment = params['alignment']

    topk_idx, num_tokens = generate_test_data(params)
    num_experts = params['num_experts']

    func = lambda: tile_kernels.moe.get_fused_mapping(topk_idx, num_experts, 0, alignment)

    t_us = benchmark_timer(func)
    result = func()
    num_bytes = count_bytes(topk_idx, *result[:7])
    bandwidth_gbs = num_bytes / t_us / 1e3

    params.pop('num_send_tokens')
    benchmark_record(
        kernel='get_fused_mapping',
        operation='fwd',
        params={'num_tokens': num_tokens, **params},
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )

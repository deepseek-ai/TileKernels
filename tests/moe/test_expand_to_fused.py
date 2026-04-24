import os
import torch

import pytest

import tile_kernels
from tile_kernels.testing.generator import generate_topk_idx, generate_hidden_sizes, generate_moe_params
from tile_kernels.testing.numeric import assert_equal, count_bytes
from tile_kernels.testing.bench import make_param_id
from tests.conftest import IS_HIP

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'

# expand_to_fused depends on get_fused_mapping which uses T.sync_warp() not supported on HIP/AMD targets
pytestmark = pytest.mark.skipif(IS_HIP, reason='expand_to_fused depends on get_fused_mapping (T.sync_warp()) not supported on HIP/AMD')


def generate_test_data_expand_to_fused(params):
    num_experts = params['num_experts']
    num_ep_ranks = params['num_ep_ranks']
    hidden = params['hidden']

    topk_idx = generate_topk_idx(params)
    num_tokens = topk_idx.shape[0]
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    pos_to_expert, _, _, token_topk_to_pos, _, _, _, _ = tile_kernels.moe.get_fused_mapping(topk_idx, num_experts, 0, 16)

    return (x, pos_to_expert, token_topk_to_pos, num_tokens)


def generate_test_params_expand(is_benchmark: bool) -> list[dict]:
    return [
        {**moe, 'hidden': hidden}
        for moe in generate_moe_params(is_benchmark=is_benchmark)
        for hidden in generate_hidden_sizes()
    ]


@pytest.mark.parametrize('params', generate_test_params_expand(is_benchmark=False), ids=make_param_id)
def test_expand_to_fused(params):
    x, pos_to_expert, token_topk_to_pos, num_tokens = generate_test_data_expand_to_fused(params)

    expanded_x = tile_kernels.moe.expand_to_fused(x, token_topk_to_pos, pos_to_expert)

    # Test correctness: torch reference
    expanded_x_ref = tile_kernels.torch.expand_to_fused(x, token_topk_to_pos, pos_to_expert)
    assert_equal(expanded_x, expanded_x_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params_expand(is_benchmark=True), ids=make_param_id)
def test_expand_to_fused_benchmark(benchmark_timer, benchmark_record, params):
    x, pos_to_expert, token_topk_to_pos, num_tokens = generate_test_data_expand_to_fused(params)

    expanded_x = tile_kernels.moe.expand_to_fused(x, token_topk_to_pos, pos_to_expert)

    t_us = benchmark_timer(lambda: tile_kernels.moe.expand_to_fused(x, token_topk_to_pos, pos_to_expert))

    num_bytes = count_bytes(x, token_topk_to_pos, pos_to_expert, expanded_x)
    bandwidth_gbs = num_bytes / t_us / 1e3

    params.pop('num_send_tokens')
    benchmark_record(
        kernel='expand_to_fused',
        operation='fwd',
        params={'num_tokens': num_tokens, **params},
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )


def generate_test_data_expand_to_fused_with_sf(params):
    num_experts = params['num_experts']
    num_ep_ranks = params['num_ep_ranks']
    hidden = params['hidden']
    num_per_channels = params['num_per_channels']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']
    round_sf = params['round_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']

    topk_idx = generate_topk_idx(params)
    num_tokens = topk_idx.shape[0]
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    x_fp8, x_sf = tile_kernels.quant.per_token_cast(
        x, 'e4m3',
        num_per_channels=num_per_channels,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )
    pos_to_expert, _, _, token_topk_to_pos, _, _, _, _ = tile_kernels.moe.get_fused_mapping(topk_idx, num_experts, 0, 16)

    return (x_fp8, x_sf, pos_to_expert, token_topk_to_pos, num_tokens)


def generate_test_params_expand_with_sf(is_benchmark: bool) -> list[dict]:
    return [
        {**moe, 'hidden': hidden, 'num_per_channels': num_per_channels,
         'use_tma_aligned_col_major_sf': col_major, 'round_sf': round_sf,
         'use_packed_ue8m0': packed_ue8m0}
        for moe in generate_moe_params(is_benchmark=is_benchmark)
        for hidden in generate_hidden_sizes()
        for num_per_channels in (32, 128)
        for col_major, round_sf, packed_ue8m0 in (
            [(False, True, False)] if IS_HIP else [(False, True, False), (True, True, True)]
        )
    ]


@pytest.mark.parametrize('params', generate_test_params_expand_with_sf(is_benchmark=False), ids=make_param_id)
def test_expand_to_fused_with_sf(params):
    x_fp8, x_sf, pos_to_expert, token_topk_to_pos, num_tokens = generate_test_data_expand_to_fused_with_sf(params)
    num_per_channels = params['num_per_channels']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']

    func = lambda: tile_kernels.moe.expand_to_fused_with_sf(
        (x_fp8, x_sf.contiguous()), num_per_channels, token_topk_to_pos, pos_to_expert, use_tma_aligned_col_major_sf,
    )

    expanded_x, expanded_x_sf = func()

    # Test correctness: torch reference
    expanded_x_ref, expanded_x_sf_ref = tile_kernels.torch.expand_to_fused_with_sf(
        (x_fp8, x_sf.contiguous()), num_per_channels, token_topk_to_pos, pos_to_expert, use_tma_aligned_col_major_sf,
    )
    assert_equal(expanded_x, expanded_x_ref)
    assert_equal(expanded_x_sf, expanded_x_sf_ref, check_stride=num_tokens > 0)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params_expand_with_sf(is_benchmark=True), ids=make_param_id)
def test_expand_to_fused_with_sf_benchmark(benchmark_timer, benchmark_record, params):
    x_fp8, x_sf, pos_to_expert, token_topk_to_pos, num_tokens = generate_test_data_expand_to_fused_with_sf(params)
    num_per_channels = params['num_per_channels']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']

    func = lambda: tile_kernels.moe.expand_to_fused_with_sf(
        (x_fp8, x_sf.contiguous()), num_per_channels, token_topk_to_pos, pos_to_expert, use_tma_aligned_col_major_sf,
    )

    expanded_x, expanded_x_sf = func()

    t_us = benchmark_timer(func)

    num_bytes = count_bytes(x_fp8, x_sf, token_topk_to_pos, pos_to_expert, expanded_x, expanded_x_sf)
    bandwidth_gbs = num_bytes / t_us / 1e3

    params.pop('num_send_tokens')
    benchmark_record(
        kernel='expand_to_fused_with_sf',
        operation='fwd',
        params={'num_tokens': num_tokens, **params},
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )

import os
import torch

import pytest

import tile_kernels
from tile_kernels.testing.bench import dtype_to_str, make_param_id
from tile_kernels.testing.generator import generate_topk_idx, generate_hidden_sizes, generate_moe_params
from tile_kernels.testing.numeric import assert_equal, count_bytes
from tests.conftest import IS_HIP

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'

# reduce_fused depends on get_fused_mapping which uses __match_any_sync (no AMD equivalent)
pytestmark = pytest.mark.skipif(IS_HIP, reason='reduce_fused depends on get_fused_mapping which uses __match_any_sync (no HIP/AMD equivalent)')



def generate_test_data(params):
    hidden = params['hidden']
    with_weights = params['with_weights']
    in_dtype = params['in_dtype']
    out_dtype = params['out_dtype']
    with_sf = params['with_sf']
    num_experts = params['num_experts']
    num_ep_ranks = params['num_ep_ranks']
    num_topk = params['num_topk']

    topk_idx = generate_topk_idx(params)
    num_tokens = topk_idx.shape[0]
    num_expanded_tokens = num_tokens * num_topk
    expanded = torch.randn((num_expanded_tokens, hidden), dtype=in_dtype, device='cuda')
    _, _, _, token_topk_to_pos, _, _, _, _ = tile_kernels.moe.get_fused_mapping(topk_idx, num_experts, 0, 1)

    topk_weights = torch.rand((num_tokens, num_topk), dtype=torch.float32, device='cuda') if with_weights else None
    if out_dtype == torch.float8_e4m3fn:
        sf = torch.randn((1,), dtype=torch.float32, device='cuda')
    else:
        sf = None
    if with_sf:
        x_sf = torch.randn((num_expanded_tokens,), dtype=torch.float32, device='cuda')
    else:
        x_sf = None
    fp8_format = 'e4m3' if out_dtype == torch.float8_e4m3fn else ''

    x_input = (expanded, x_sf) if x_sf is not None else expanded

    return (expanded, token_topk_to_pos, topk_weights, sf, x_sf, fp8_format, x_input, num_tokens)


def generate_test_params(is_benchmark: bool) -> list[dict]:
    params = [
        {**moe, 'hidden': hidden, 'with_weights': with_weights,
         'in_dtype': in_dtype, 'out_dtype': out_dtype, 'with_sf': with_sf}
        for moe in generate_moe_params(is_benchmark=is_benchmark)
        for hidden in generate_hidden_sizes(256)
        for with_weights in (True, False)
        for in_dtype in (torch.float32, torch.bfloat16)
        for out_dtype in (in_dtype, torch.float8_e4m3fn)
        for with_sf in (True, False)
    ]
    if is_benchmark:
        params = [p for p in params if p['num_topk'] == 6 and p['with_weights']]
    return params


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_reduce_fused(params):
    (expanded, token_topk_to_pos, topk_weights, sf, x_sf, fp8_format, x_input,
     _) = generate_test_data(params)

    # Test correctness: tile_kernels kernel
    func = lambda: tile_kernels.moe.reduce_fused(
        x_input, topk_weights, token_topk_to_pos, fp8_format, sf, None
    )
    r_tk = func()

    # Test correctness: torch reference
    r_ref = tile_kernels.torch.reduce_fused(
        x_input, topk_weights, token_topk_to_pos, fp8_format, sf
    )
    assert_equal(r_tk, r_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_reduce_fused_benchmark(benchmark_timer, benchmark_record, params):
    hidden = params['hidden']
    out_dtype = params['out_dtype']

    (expanded, token_topk_to_pos, topk_weights, sf, x_sf, fp8_format, x_input,
     num_tokens) = generate_test_data(params)
    in_dtype = params['in_dtype']

    func = lambda: tile_kernels.moe.reduce_fused(
        x_input, topk_weights, token_topk_to_pos, fp8_format, sf, None
    )
    r_tk = func()

    num_bytes = count_bytes(token_topk_to_pos, x_sf, r_tk)
    num_bytes += torch.count_nonzero(token_topk_to_pos != -1).item() * hidden * (torch.finfo(in_dtype).bits // 8)
    if topk_weights is not None:
        num_bytes += count_bytes(topk_weights)

    t_us = benchmark_timer(func)

    bandwidth_gbs = num_bytes / t_us / 1e3

    params.pop('num_send_tokens')
    benchmark_record(
        kernel='reduce_fused',
        operation='fwd',
        params={'num_tokens': num_tokens, **params, 'in_dtype': dtype_to_str(in_dtype), 'out_dtype': dtype_to_str(out_dtype)},
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )

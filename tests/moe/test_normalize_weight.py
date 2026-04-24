import os
import torch

import pytest

import tile_kernels
from tile_kernels.testing.generator import generate_topk_idx, generate_moe_params
from tile_kernels.testing.numeric import assert_equal, count_bytes
from tile_kernels.torch import normalize_weight as torch_normalize_weight
from tile_kernels.testing.bench import make_param_id
from tests.conftest import IS_HIP

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'

# normalize_weight_kernel produces incorrect results on HIP/AMD (NaN outputs) due to
# HIP-incompatible T.vectorized usage in the kernel implementation
pytestmark = pytest.mark.skipif(IS_HIP, reason='normalize_weight_kernel produces incorrect results on HIP/AMD targets')


def generate_test_data(params):
    num_topk = params['num_topk']

    topk_idx = generate_topk_idx(params)
    num_tokens = topk_idx.shape[0]
    topk_weights = torch.rand((num_tokens, num_topk), dtype=torch.float32, device='cuda')

    return (topk_weights, num_tokens)


@pytest.mark.parametrize('params', list(generate_moe_params(is_benchmark=False)), ids=make_param_id)
def test_normalize_weight(params):
    (topk_weights, _) = generate_test_data(params)

    denominator, normalized_weights = tile_kernels.moe.normalize_weight(topk_weights)

    # Test correctness: torch reference
    denom_ref, norm_ref = torch_normalize_weight(topk_weights)
    assert_equal(denominator, denom_ref)
    assert_equal(normalized_weights, norm_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', list(generate_moe_params(is_benchmark=True)), ids=make_param_id)
def test_normalize_weight_benchmark(benchmark_timer, benchmark_record, params):
    topk_weights, num_tokens = generate_test_data(params)
    num_topk = params['num_topk']

    denominator, normalized_weights = tile_kernels.moe.normalize_weight(topk_weights)

    t_us = benchmark_timer(lambda: tile_kernels.moe.normalize_weight(topk_weights))
    num_bytes = count_bytes(topk_weights, denominator, normalized_weights)
    bandwidth_gbs = num_bytes / t_us / 1e3

    params.pop('num_send_tokens')
    benchmark_record(
        kernel='normalize_weight',
        operation='fwd',
        params={'num_tokens': num_tokens, **params, 'num_topk': num_topk},
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )

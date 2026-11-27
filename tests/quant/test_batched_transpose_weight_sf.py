import os
import pytest
import torch

import tile_kernels
from tile_kernels.testing.generator import generate_rand_float
from tile_kernels.testing.numeric import assert_equal, count_bytes

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


_MIXED_WEIGHT_CONFIGS = [
    (128, 128, None, False),
    (384, 512, 2, True),
    (768, 1152, 3, False),
    (640, 1280, 1, True),
    (2048, 2560, 2, False),
    (384, 384, None, True),
    (6144, 4096, None, False),
]


_CASES = [
    pytest.param(32, _MIXED_WEIGHT_CONFIGS, id='block32-mixed'),
]


def generate_test_data(block_size, weight_configs):
    sf, transpose_sf1s = [], []
    for n, k, num_groups, transpose_sf1 in weight_configs:
        weight = generate_rand_float(((num_groups or 1) * n, k))
        weight_sf = tile_kernels.quant.per_block_cast_with_sf_only(weight, 'e4m3', (block_size, block_size), round_sf=True)
        sf.append(weight_sf.view(num_groups, n // block_size, k // block_size) if num_groups else weight_sf)
        transpose_sf1s.append(transpose_sf1)
    return sf, transpose_sf1s


@pytest.mark.parametrize('block_size,weight_configs', _CASES)
def test_batched_transpose_weight_sf(block_size, weight_configs):
    sf, transpose_sf1s = generate_test_data(block_size, weight_configs)

    outs = [None] * len(weight_configs)
    for _ in range(2):
        sf_pairs = tile_kernels.quant.batched_transpose_weight_sf(sf, block_size, transpose_sf1s, outs)
        ref_pairs = tile_kernels.torch.batched_transpose_weight_sf(sf, block_size, transpose_sf1s)
        assert len(sf_pairs) == len(weight_configs)
        for out, (sf0_out, sf1_out), (sf0_ref, sf1_ref) in zip(outs, sf_pairs, ref_pairs):
            if out is not None:
                assert sf0_out.data_ptr() == out[0].data_ptr() and sf1_out.data_ptr() == out[1].data_ptr()
            assert_equal(sf0_out, sf0_ref)
            assert_equal(sf1_out, sf1_ref)
            assert sf0_out.data_ptr() % 32 == 0 and sf1_out.data_ptr() % 32 == 0

        new_sf, _ = generate_test_data(block_size, weight_configs)
        torch._foreach_copy_(sf, new_sf)
        outs = [None, *sf_pairs[1:]]


@pytest.mark.benchmark
@pytest.mark.parametrize('block_size,weight_configs', _CASES)
def test_batched_transpose_weight_sf_benchmark(block_size, weight_configs, benchmark_timer, benchmark_record):
    sf, transpose_sf1s = generate_test_data(block_size, weight_configs)

    sf_pairs = tile_kernels.quant.batched_transpose_weight_sf(sf, block_size, transpose_sf1s, [None] * len(weight_configs))
    func = lambda: tile_kernels.quant.batched_transpose_weight_sf(sf, block_size, transpose_sf1s, sf_pairs)
    t_us = benchmark_timer(func)
    num_bytes = count_bytes(sf, sf_pairs)

    benchmark_record(
        kernel='batched_transpose_weight_sf',
        operation='fwd',
        params={'block_size': block_size, 'num_weights': len(weight_configs)},
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )

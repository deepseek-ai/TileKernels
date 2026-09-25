import os

import pytest
import torch

import tile_kernels
from tile_kernels.testing import clear_unused_sf
from tile_kernels.testing.bench import make_param_id
from tile_kernels.testing.numeric import assert_equal, count_bytes


# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    rstd = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
    return x_float * rstd * weight.float()


def generate_test_params(is_benchmark: bool) -> list[dict]:
    if is_benchmark:
        return [{
            'num_tokens': 4001,
            'hidden': 7168,
            'in_dtype': torch.bfloat16,
            'num_per_channels': 128,
            'use_tma_aligned_col_major_sf': True,
            'round_sf': True,
            'use_packed_ue8m0': True,
        }]

    return [
        {
            'num_tokens': num_tokens,
            'hidden': hidden,
            'in_dtype': in_dtype,
            'num_per_channels': num_per_channels,
            'use_tma_aligned_col_major_sf': use_tma_aligned_col_major_sf,
            'round_sf': round_sf,
            'use_packed_ue8m0': use_packed_ue8m0,
        }
        for num_tokens, hidden, in_dtype, num_per_channels in [
            (0, 512, torch.bfloat16, 128),
            (17, 512, torch.bfloat16, 128),
            (17, 512, torch.float32, 128),
            (17, 512, torch.float32, 512),
            (17, 640, torch.bfloat16, 128),
            (17, 7168, torch.bfloat16, 128),
        ]
        for use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0 in [
            (False, False, False),
            (False, True, False),
            (True, True, True),
        ]
        if num_per_channels != hidden or not use_tma_aligned_col_major_sf
    ]


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_rmsnorm_forward_and_per_token_cast(params):
    torch.manual_seed(0)
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    in_dtype = params['in_dtype']
    eps = 1e-6

    x = torch.randn((num_tokens, hidden), dtype=in_dtype, device='cuda')
    weight = torch.randn((hidden,), dtype=in_dtype, device='cuda')
    base_args = {
        'fmt': 'e4m3',
        'num_per_channels': params['num_per_channels'],
        'use_tma_aligned_col_major_sf': params['use_tma_aligned_col_major_sf'],
        'round_sf': params['round_sf'],
        'use_packed_ue8m0': params['use_packed_ue8m0'],
    }

    out, out_sf = tile_kernels.quant.rmsnorm_forward_and_per_token_cast(
        x=x,
        weight=weight,
        eps=eps,
        **base_args,
    )
    out_ref, out_sf_ref = tile_kernels.torch.cast(
        rmsnorm_ref(x, weight, eps),
        fmt=base_args['fmt'],
        block_size=(1, params['num_per_channels']),
        use_tma_aligned_col_major_sf=base_args['use_tma_aligned_col_major_sf'],
        round_sf=base_args['round_sf'],
        use_packed_ue8m0=base_args['use_packed_ue8m0'],
    )

    if params['use_packed_ue8m0']:
        out_sf = clear_unused_sf(out_sf, hidden, params['num_per_channels'])
        out_sf_ref = clear_unused_sf(out_sf_ref, hidden, params['num_per_channels'])

    # The RMS reduction order differs from PyTorch and can move values on an FP8 rounding boundary.
    num_mismatched = torch.count_nonzero(out != out_ref).item()
    assert num_mismatched <= max(1, out.numel() // 1000)
    torch.testing.assert_close(out.float(), out_ref.float(), rtol=0.125, atol=1.0)

    if out_sf.dtype == torch.float32:
        torch.testing.assert_close(out_sf, out_sf_ref, rtol=2e-6, atol=0)
    else:
        assert_equal(out_sf, out_sf_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_rmsnorm_forward_and_per_token_cast_benchmark(benchmark_timer, benchmark_record, params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    in_dtype = params['in_dtype']
    eps = 1e-6
    x = torch.randn((num_tokens, hidden), dtype=in_dtype, device='cuda')
    weight = torch.randn((hidden,), dtype=in_dtype, device='cuda')

    func = lambda: tile_kernels.quant.rmsnorm_forward_and_per_token_cast(
        x=x,
        weight=weight,
        eps=eps,
        fmt='e4m3',
        num_per_channels=params['num_per_channels'],
        use_tma_aligned_col_major_sf=params['use_tma_aligned_col_major_sf'],
        round_sf=params['round_sf'],
        use_packed_ue8m0=params['use_packed_ue8m0'],
    )
    out, out_sf = func()
    t_us = benchmark_timer(func)

    benchmark_record(
        kernel='rmsnorm_forward_and_per_token_cast',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=count_bytes(x, weight, out, out_sf) / t_us / 1e3,
    )

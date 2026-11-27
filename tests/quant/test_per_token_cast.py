import os
import math
import pytest
import torch

import tile_kernels
from tile_kernels.rand import randn
from tile_kernels.config import set_amax_clamp_for_quant, get_amax_clamp_for_quant, is_ascend, get_device
from tile_kernels.testing.bench import make_param_id
from tile_kernels.utils import str_to_dtype
from tile_kernels.testing.numeric import assert_equal, count_bytes, check_bias
from tile_kernels.testing.generator import generate_hidden_sizes, generate_num_tokens, get_test_level
from tile_kernels.testing.quant import clear_unused_sf, quant_level_rank

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_test_data(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    in_dtype = params['in_dtype']
    fmt = params['fmt']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']
    round_sf = params['round_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']
    use_e4m3_sf = params['use_e4m3_sf']
    x_block_size = params.get('x_block_size')

    in_with_sf_factor = in_dtype in ('e4m3', 'e2m1')

    if in_with_sf_factor:
        x = randn((num_tokens, hidden), dtype=torch.bfloat16, device=get_device())
        original_x = x
        x = tile_kernels.torch.cast(
            x,
            in_dtype,
            x_block_size,
            use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
            round_sf=round_sf,
            use_packed_ue8m0=use_packed_ue8m0,
        )
    else:
        x = randn((num_tokens, hidden), dtype=str_to_dtype(in_dtype), device=get_device())
        original_x = x

    base_args = dict(
        x=x,
        fmt=fmt,
        x_block_size=x_block_size,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
        use_e4m3_sf=use_e4m3_sf,
    )

    return (x, original_x, base_args, in_with_sf_factor)


def generate_test_params(level: int) -> list[dict]:
    hidden_sizes = generate_hidden_sizes(
        level, ['main', 'ffn', '2*ffn', 'q_lora', '128*indexer', 'wv', 'head', 'attn_context', 'engram_head', 'engram_input', '5*main'], align=128
    )
    if level == 0:
        # Too many test cases, take some representative ones on level 0
        hidden_sizes = hidden_sizes[::4]
    params = [
        {
            'num_tokens': num_tokens,
            'hidden': hidden_size,
            'use_tma_aligned_col_major_sf': use_tma_aligned_col_major_sf,
            'round_sf': round_sf,
            'use_packed_ue8m0': use_packed_ue8m0,
            'use_e4m3_sf': use_e4m3_sf,
            'in_dtype': in_dtype,
            'num_per_channels': num_per_channels,
            'x_block_size': x_block_size,
            'fmt': fmt,
        }
        for num_tokens in generate_num_tokens(level)
        for hidden_size in hidden_sizes
        for use_tma_aligned_col_major_sf in ([True] if level == 0 else [False, True])
        for use_packed_ue8m0 in ([True] if level == 0 else [False, True])
        for round_sf in ([True] if use_packed_ue8m0 else [True, False])
        for in_dtype in ['fp32', 'bf16', 'e4m3', 'e2m1']
        for num_per_channels in ((32,) if is_ascend() or in_dtype in ('e4m3', 'e2m1') else (16, 32))
        for use_e4m3_sf in ((False,) if (round_sf or is_ascend()) else (False, True))
        for x_block_size in (((32, 32),) if in_dtype in ('e4m3', 'e2m1') else (None,))
        for fmt in ('e4m3', 'e2m1')
    ]
    return params


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_per_token_cast(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']
    round_sf = params['round_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']
    use_e4m3_sf = params['use_e4m3_sf']
    in_dtype = params['in_dtype']
    num_per_channels = params['num_per_channels']
    x_block_size = params.get('x_block_size')
    fmt = params['fmt']

    in_with_sf_factor = in_dtype in ('e4m3', 'e2m1')
    # Test correctness
    x, original_x, base_args, in_with_sf_factor = generate_test_data(params)
    func = lambda: tile_kernels.quant.per_token_cast(
        **base_args,
        num_per_channels=num_per_channels,
    )
    func_ref = lambda: tile_kernels.torch.cast(
        **base_args,
        block_size=(1, num_per_channels),
    )
    x_casted, x_sf = func()
    x_casted_ref, x_sf_ref = func_ref()
    x_casted_back = tile_kernels.torch.cast_back((x_casted, x_sf), 'fp32', (1, num_per_channels))

    if use_packed_ue8m0:
        x_sf = clear_unused_sf(x_sf, hidden, num_per_channels)
        x_sf_ref = clear_unused_sf(x_sf_ref, hidden, num_per_channels)

    assert_equal(x_casted, x_casted_ref)
    assert_equal(x_sf, x_sf_ref)

    # Check bias
    check_bias(x_casted_back, original_x)

    # Test non-contiguous input (stride(0) != hidden)
    if not in_with_sf_factor and fmt == 'e4m3' and not use_packed_ue8m0 and num_tokens > 0:
        x_non_contiguous = randn((num_tokens, hidden * 2), dtype=str_to_dtype(in_dtype), device=get_device())[:, :hidden]
        x_non_contiguous.copy_(original_x)
        x_non_contiguous_casted, x_non_contiguous_sf = tile_kernels.quant.per_token_cast(
            x=x_non_contiguous,
            fmt=fmt,
            x_block_size=x_block_size,
            use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
            round_sf=round_sf,
            use_packed_ue8m0=use_packed_ue8m0,
            use_e4m3_sf=use_e4m3_sf,
            num_per_channels=num_per_channels,
        )
        assert_equal(x_non_contiguous_casted, x_casted)
        assert_equal(x_non_contiguous_sf, x_sf)

    if not in_with_sf_factor and num_per_channels != hidden:
        base_args.pop('x_block_size')
        # Precomputed scales make round_sf irrelevant; test cast only once with True.
        if round_sf and not use_tma_aligned_col_major_sf and not use_packed_ue8m0:
            # TMA aligned or packed ue8m0 sf is used for FP8/FP4 GEMM, not for other cast
            x_casted = tile_kernels.quant.per_token_cast_with_precomputed_sf(**base_args, num_per_channels=num_per_channels, sf=x_sf)
            x_casted_ref = tile_kernels.torch.cast(**base_args, block_size=(1, num_per_channels), sf=x_sf)
            assert_equal(x_casted, x_casted_ref)

        # Test sf only mode
        twice_sf_inv = tile_kernels.quant.per_token_cast_with_sf_only(**base_args, num_per_channels=num_per_channels)
        if use_packed_ue8m0:
            twice_sf_inv = clear_unused_sf(twice_sf_inv, hidden, num_per_channels)
        assert_equal(twice_sf_inv, x_sf)


def test_set_amax_clamp_for_quant():
    num_tokens, hidden = 32, 128
    default_clamp = get_amax_clamp_for_quant('e4m3', False)
    x = randn((num_tokens, hidden), dtype=torch.bfloat16, device=get_device())
    _, sf_default = tile_kernels.quant.per_token_cast(x, 'e4m3', num_per_channels=32, round_sf=False)
    set_amax_clamp_for_quant('e4m3', False, 1e6)
    try:
        x_casted, x_sf = tile_kernels.quant.per_token_cast(x, 'e4m3', num_per_channels=32, round_sf=False)
        x_casted_ref, x_sf_ref = tile_kernels.torch.cast(x, 'e4m3', block_size=(1, 32), round_sf=False)
        assert_equal(x_casted, x_casted_ref)
        assert_equal(x_sf, x_sf_ref)
        min_expected_sf = 1e6 / 448.0
        assert (x_sf >= min_expected_sf * 0.99).all()
        assert (x_sf > sf_default).all()
    finally:
        set_amax_clamp_for_quant('e4m3', False, default_clamp)


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_per_token_cast_stochastic(params):
    if is_ascend() and params['fmt'] != 'e4m3':
        pytest.skip('Ascend per_token_cast with stochastic rounding only supports E4M3 output')
    hidden = params['hidden']
    num_per_channels = params['num_per_channels']
    use_packed_ue8m0 = params['use_packed_ue8m0']
    fmt = params['fmt']

    x, original_x, base_args, in_with_sf = generate_test_data(params)
    if in_with_sf:
        ref = tile_kernels.torch.cast_back(x, 'fp32', params['x_block_size'])
    else:
        ref = original_x.float()
    scale = ref.abs().mean()

    func_stochastic = lambda: tile_kernels.quant.per_token_cast(**base_args, num_per_channels=num_per_channels, stochastic_cast=True)
    func_nearest = lambda: tile_kernels.quant.per_token_cast(**base_args, num_per_channels=num_per_channels, stochastic_cast=False)

    if ref.shape[0] == 0:
        func_stochastic()
        return

    # Check determinism
    x1_casted, sf1 = func_stochastic()
    x2_casted, sf2 = func_stochastic()
    sf1_masked, sf2_masked = sf1, sf2
    if use_packed_ue8m0:
        sf1_masked = clear_unused_sf(sf1, hidden, num_per_channels)
        sf2_masked = clear_unused_sf(sf2, hidden, num_per_channels)
    assert_equal(x1_casted, x2_casted)
    assert_equal(sf1_masked, sf2_masked)

    # Check SR differs from RNE (skipped if input is exactly representable in the output format)
    x_casted_nearest, sf_nearest = func_nearest()
    in_dtype = params['in_dtype']
    exactly_representable = in_dtype == 'e2m1' or (in_dtype == 'e4m3' and fmt == 'e4m3')
    if not exactly_representable:
        assert not torch.equal(x1_casted.view(torch.uint8), x_casted_nearest.view(torch.uint8)), 'SR output is identical to RNE'

    # Check rounding direction bias
    dequant_stochastic = tile_kernels.torch.cast_back((x1_casted, sf1), 'fp32', (1, num_per_channels))
    check_bias(dequant_stochastic, ref)

    # Check rounding magnitude bias: mean signed error ~ N(0, step^2 / n), step ~ 2^(-mantissa_bits)
    mantissa_bits = 3 if fmt == 'e4m3' else 1
    step = 2.0 ** (-mantissa_bits)
    mean_error = (dequant_stochastic - ref).mean().abs() / scale
    allowed_mean_error = 10 * step / math.sqrt(ref.numel())
    assert mean_error < allowed_mean_error, f'Mean error not close to 0 (size = {ref.numel()}): {mean_error=:.2e} {allowed_mean_error=:.2e}'

    # Check SR and RNE pick numerically adjacent quant levels, and use identical sf.
    sf_nearest_masked = clear_unused_sf(sf_nearest, hidden, num_per_channels) if use_packed_ue8m0 else sf_nearest
    assert_equal(sf1_masked, sf_nearest_masked)
    rank_stochastic = quant_level_rank(x1_casted, fmt)
    rank_nearest = quant_level_rank(x_casted_nearest, fmt)
    assert ((rank_stochastic - rank_nearest).abs() <= 1).all(), (
        f'SR output is not adjacent to RNE: max rank gap={(rank_stochastic - rank_nearest).abs().max().item()}'
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0), ids=make_param_id)
def test_per_token_cast_benchmark(benchmark_timer, benchmark_record, params):
    num_per_channels = params['num_per_channels']

    x, _, base_args, _ = generate_test_data(params)
    func = lambda: tile_kernels.quant.per_token_cast(
        **base_args,
        num_per_channels=num_per_channels,
    )
    x_casted, x_sf = func()

    t_us = benchmark_timer(func)
    num_bytes = count_bytes(x, x_casted, x_sf)

    benchmark_record(
        kernel='per_token_cast',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0), ids=make_param_id)
def test_per_token_cast_stochastic_benchmark(benchmark_timer, benchmark_record, params):
    if is_ascend() and params['fmt'] != 'e4m3':
        pytest.skip('Ascend per_token_cast with stochastic rounding only supports E4M3 output')
    num_per_channels = params['num_per_channels']

    x, _, base_args, _ = generate_test_data(params)
    func = lambda: tile_kernels.quant.per_token_cast(
        **base_args,
        num_per_channels=num_per_channels,
        stochastic_cast=True,
    )
    x_casted, x_sf = func()

    t_us = benchmark_timer(func)
    num_bytes = count_bytes(x, x_casted, x_sf)

    benchmark_record(
        kernel='per_token_cast_stochastic',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )

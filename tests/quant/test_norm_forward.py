import os
import pytest
import torch

import tile_kernels
from tile_kernels.config import get_device
from tile_kernels.rand import randn
from tile_kernels.testing import clear_unused_sf
from tile_kernels.testing.bench import get_cast_params, make_param_id
from tile_kernels.testing.generator import (
    generate_samples,
    generate_num_tokens,
    generate_hidden_sizes,
    generate_cast_config,
    get_test_level,
)
from tile_kernels.testing.numeric import assert_equal, count_bytes
from tile_kernels.torch import cast, cast_back, norm_forward_ref
from tile_kernels.utils import align, ceil_div, str_to_dtype

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_test_data(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    dtype = str_to_dtype(params['in_dtype'] or params['fmt'])
    batch_size = params['batch_size']
    width = align(hidden, 16) + params['row_padding']
    # Batched inputs allocate one extra row so the batch stride exceeds the sliced extent.
    shape = (num_tokens, width) if batch_size is None else (batch_size, num_tokens + 1, width)
    x = randn(shape, dtype=dtype, device=get_device())[..., :num_tokens, :hidden]
    weight = randn((hidden,), dtype=dtype, device=x.device) if params['use_weight'] else None
    residual = randn(x.shape, dtype=dtype, device=x.device) if params['has_residual'] else None
    mega_moe_sf = None
    if params['mega_moe_block_m']:
        rows = ceil_div(num_tokens, params['mega_moe_block_m']) * align(params['mega_moe_block_m'], 128)
        mega_moe_sf = torch.full((hidden // 128, rows), -1, dtype=torch.int32, device=x.device).mT
    return x, weight, residual, mega_moe_sf


def generate_test_params(level: int) -> list[dict]:
    params = [
        {
            **args,
            'fmt': fmt,
            'in_dtype': in_dtype,
            'hidden': hidden,
            'num_tokens': num_tokens,
            'use_weight': use_weight,
            'num_per_channels': num_per_channels,
            'use_tma_aligned_col_major_sf': use_tma_aligned_col_major_sf,
            'round_sf': round_sf,
            'use_packed_ue8m0': use_packed_ue8m0,
            'eps': eps,
        }
        for fmt in ('bf16', 'fp32', 'e4m3')
        for in_dtype in (('bf16', 'fp32') if fmt == 'e4m3' else (None,))
        # The kernels only support the DeepGEMM SF granularity of 32 channels
        for num_per_channels in ((32,) if fmt == 'e4m3' else (None,))
        for hidden in ([4096, 5120, 7168] if level == 0 else generate_hidden_sizes())
        if num_per_channels is None or hidden % num_per_channels == 0
        for num_tokens in generate_num_tokens(level)
        for eps in (1e-6,)
        for use_weight in (True, False)
        for use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0 in generate_cast_config(level, fmt)
        for args in generate_samples(
            level,
            has_residual=(False, True),
            batch_size=(None,) if level == 0 else (None, 2),
            out_scale=(1.0, 0.3),
            row_padding=(0, 16),
            with_out_bf16=(True, False) if fmt == 'e4m3' else (False,),
            mega_moe_block_m=(192, 0) if fmt == 'e4m3' and round_sf and hidden % 128 == 0 else (0,),
        )
        if not args['mega_moe_block_m'] or args['batch_size'] is None
    ]
    return params


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_norm_forward_and_per_token_cast(params):
    fmt = params['fmt']
    hidden = params['hidden']
    eps = params['eps']
    out_scale = params['out_scale']
    num_per_channels = params['num_per_channels']
    with_out_bf16 = params['with_out_bf16']
    use_packed_ue8m0 = params['use_packed_ue8m0']
    mega_moe_block_m = params['mega_moe_block_m']

    x, weight, residual, mega_moe_sf = generate_test_data(params)
    out = torch.empty_strided(x.shape, x.stride(), dtype=torch.bfloat16 if fmt == 'e4m3' else x.dtype, device=x.device)
    residual_out = torch.empty_like(x) if residual is not None else None
    args = dict(x=x, weight=weight, eps=eps, residual=residual, out_scale=out_scale, residual_out=residual_out)
    if fmt == 'e4m3':
        func = lambda quant_out=None: tile_kernels.quant.norm_forward_and_per_token_cast(
            **args,
            fmt=fmt,
            num_per_channels=num_per_channels,
            **get_cast_params(params),
            with_out_bf16=with_out_bf16,
            out=quant_out,
            out_bf16=out if with_out_bf16 else None,
            mega_moe_sf=mega_moe_sf[: x.shape[0]] if mega_moe_sf is not None else None,
            mega_moe_block_m=mega_moe_block_m,
        )
    else:
        func = lambda: tile_kernels.quant.norm_forward(**args, out=out)

    out_ref, rstd_ref, norm_input_ref = norm_forward_ref(x, weight, eps, residual, out_scale)

    def check_norm_outputs(norm_out, rstd, norm_input):
        assert norm_out is (out if fmt != 'e4m3' or with_out_bf16 else None)
        if residual_out is not None:
            assert norm_input is residual_out
        assert_equal(norm_input, norm_input_ref, check_stride=False)
        if norm_out is not None:
            torch.testing.assert_close(norm_out, out_ref.to(norm_out.dtype))
        torch.testing.assert_close(rstd, rstd_ref, rtol=2e-6, atol=1e-7)

    # Test the norm outputs
    if fmt == 'e4m3':
        (y, sf), norm_out, rstd, norm_input = func()
    else:
        norm_out, rstd, norm_input = func()
    check_norm_outputs(norm_out, rstd, norm_input)

    if fmt != 'e4m3':
        return

    # Test the quantized outputs. FP32 quantization precedes the optional BF16 output conversion.
    normalized = norm_out
    if normalized is None or x.dtype == torch.float32:
        normalized, _, _ = tile_kernels.quant.norm_forward(x, weight, eps, residual, out_scale)
        torch.testing.assert_close(normalized, out_ref)
    for output, output_sf, unquantized, out_ref_row in (
        zip(y.unsqueeze(0), sf.unsqueeze(0), normalized.unsqueeze(0), out_ref.unsqueeze(0)) if x.ndim == 2 else zip(y, sf, normalized, out_ref)
    ):
        # Loose dequantized check first, independent of the kernel-derived reference below
        dequantized = cast_back((output, output_sf), 'fp32', (1, num_per_channels))
        torch.testing.assert_close(dequantized, out_ref_row.float(), rtol=0.08, atol=0.02)
        # Cast the rounded kernel output to check quantization independently of reduction rounding
        y_ref, sf_ref = cast(unquantized, fmt, (1, num_per_channels), **get_cast_params(params))
        assert_equal(output, y_ref, check_stride=False)
        if use_packed_ue8m0:
            output_sf = clear_unused_sf(output_sf, hidden, num_per_channels)
            sf_ref = clear_unused_sf(sf_ref, hidden, num_per_channels)
        assert_equal(output_sf, sf_ref, check_stride=False)

    # Test the Mega MoE SF output
    if mega_moe_sf is not None:
        _, packed_sf = cast(normalized, fmt, (1, num_per_channels), round_sf=True, use_packed_ue8m0=True)
        token = torch.arange(x.shape[0], device=x.device)
        block_m = mega_moe_block_m
        i = token % block_m
        # Row mapping of `get_mega_moe_sf_row` in the kernels
        row = token // block_m * align(block_m, 128) + i // 128 * 128 + i % 32 * 4 + i % 128 // 32
        sf_ref = torch.full_like(mega_moe_sf, -1)
        sf_ref[row] = packed_sf.contiguous().view(torch.int32)
        assert_equal(mega_moe_sf, sf_ref, check_stride=False)

    # Test preallocated quantized outputs
    quant_out = (
        torch.empty_strided(y.shape, x.stride(), dtype=y.dtype, device=y.device),
        None if params['use_tma_aligned_col_major_sf'] else torch.empty_like(sf),
    )
    (preallocated_y, preallocated_sf), norm_out, rstd, norm_input = func(quant_out)
    check_norm_outputs(norm_out, rstd, norm_input)
    assert preallocated_y is quant_out[0]
    if quant_out[1] is not None:
        # May be returned as a packed-dtype view, so compare storage instead of identity
        assert preallocated_sf.data_ptr() == quant_out[1].data_ptr()
    assert_equal(preallocated_y, y, check_stride=False)
    if use_packed_ue8m0:
        preallocated_sf = clear_unused_sf(preallocated_sf, hidden, num_per_channels)
        sf = clear_unused_sf(sf, hidden, num_per_channels)
    assert_equal(preallocated_sf, sf, check_stride=False)


@pytest.mark.parametrize(
    'dtype,use_tma_aligned_col_major_sf,round_sf,use_packed_ue8m0',
    [
        (torch.bfloat16, False, False, False),
        (torch.float32, True, True, True),
    ],
)
def test_norm_forward_and_per_token_cast_group_128(
    dtype,
    use_tma_aligned_col_major_sf,
    round_sf,
    use_packed_ue8m0,
):
    torch.manual_seed(0)
    x = torch.randn((17, 640), dtype=dtype, device='cuda')
    residual = torch.randn_like(x)
    residual_out = torch.empty_like(x)
    weight = torch.randn((640,), dtype=dtype, device='cuda')
    expected_normalized, _, expected_residual = norm_forward_ref(x, weight, 1e-6, residual)

    quant_out, _, _, norm_input = tile_kernels.quant.norm_forward_and_per_token_cast(
        x,
        weight,
        1e-6,
        'e4m3',
        128,
        residual=residual,
        residual_out=residual_out,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )
    expected_quant = cast(
        expected_normalized,
        'e4m3',
        (1, 128),
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )

    assert norm_input is residual_out
    assert_equal(norm_input, expected_residual)
    assert_equal(quant_out[0], expected_quant[0])
    if use_packed_ue8m0:
        quant_out = (quant_out[0], clear_unused_sf(quant_out[1], 640, 128))
        expected_quant = (expected_quant[0], clear_unused_sf(expected_quant[1], 640, 128))
    assert_equal(quant_out[1], expected_quant[1])


ADD_RMSNORM_CASES = [
    (0, 512, torch.bfloat16, False, False, False),
    (17, 512, torch.bfloat16, False, True, False),
    (17, 640, torch.float32, True, True, True),
    (3, 640, torch.float32, True, False, False),
]


@pytest.mark.parametrize(
    'num_tokens,hidden,dtype,use_tma_aligned_col_major_sf,round_sf,use_packed_ue8m0',
    ADD_RMSNORM_CASES,
)
def test_add_rmsnorm_forward_and_per_token_cast(
    num_tokens,
    hidden,
    dtype,
    use_tma_aligned_col_major_sf,
    round_sf,
    use_packed_ue8m0,
):
    torch.manual_seed(1)
    x = torch.randn((num_tokens, hidden), dtype=dtype, device='cuda')
    residual = torch.randn_like(x)
    weight = torch.randn((hidden,), dtype=dtype, device='cuda')
    x_before = x.clone()
    residual_before = residual.clone()
    expected_normalized, _, expected_residual = norm_forward_ref(x, weight, 1e-6, residual_before)
    cast_args = dict(
        fmt='e4m3',
        block_size=(1, 128),
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )
    expected_quant = cast(expected_normalized, **cast_args)

    quant_out = tile_kernels.quant.add_rmsnorm_forward_and_per_token_cast(
        x,
        residual,
        weight,
        1e-6,
        'e4m3',
        128,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )

    assert_equal(x, x_before)
    assert_equal(residual, expected_residual)
    assert_equal(quant_out[0], expected_quant[0])
    if use_packed_ue8m0:
        quant_out = (quant_out[0], clear_unused_sf(quant_out[1], hidden, 128))
        expected_quant = (expected_quant[0], clear_unused_sf(expected_quant[1], hidden, 128))
    assert_equal(quant_out[1], expected_quant[1])


def test_add_rmsnorm_forward_and_per_token_cast_rejects_overlapping_views():
    storage = torch.randn((3, 512), dtype=torch.bfloat16, device='cuda')
    weight = torch.ones((512,), dtype=storage.dtype, device=storage.device)
    with pytest.raises(AssertionError):
        tile_kernels.quant.add_rmsnorm_forward_and_per_token_cast(
            storage[:2], storage[1:], weight, 1e-6, 'e4m3', 128,
        )

    x = torch.randn((2, 512), dtype=torch.bfloat16, device='cuda')
    residual = torch.randn_like(x)
    with pytest.raises(AssertionError):
        tile_kernels.quant.add_rmsnorm_forward_and_per_token_cast(
            x, residual, residual.view(-1)[128:640], 1e-6, 'e4m3', 128,
        )


def test_add_rmsnorm_forward_and_per_token_cast_rejects_whole_row_scaling():
    x = torch.randn((2, 640), dtype=torch.float32, device='cuda')
    residual = torch.randn_like(x)
    weight = torch.ones((640,), dtype=x.dtype, device=x.device)

    with pytest.raises(AssertionError):
        tile_kernels.quant.add_rmsnorm_forward_and_per_token_cast(
            x, residual, weight, 1e-6, 'e4m3', 640,
        )


@pytest.mark.benchmark
def test_add_rmsnorm_forward_and_per_token_cast_benchmark(benchmark_timer, benchmark_record):
    num_tokens, hidden, eps = 4001, 7168, 1e-6
    # Zero x keeps the in-place residual stable across timed iterations without timing a reset.
    x = torch.zeros((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    wrapper_residual = torch.randn_like(x)
    direct_residual = wrapper_residual.clone()
    unfused_residual = wrapper_residual.clone()
    weight = torch.randn((hidden,), dtype=x.dtype, device=x.device)
    cast_args = dict(
        fmt='e4m3',
        num_per_channels=128,
        use_tma_aligned_col_major_sf=True,
        round_sf=True,
        use_packed_ue8m0=True,
    )

    wrapper = lambda: tile_kernels.quant.add_rmsnorm_forward_and_per_token_cast(
        x,
        wrapper_residual,
        weight,
        eps,
        **cast_args,
    )
    direct = lambda: tile_kernels.quant.norm_forward_and_per_token_cast(
        x,
        weight,
        eps,
        residual=direct_residual,
        residual_out=direct_residual,
        **cast_args,
    )[0]

    def unfused():
        normalized, _, _ = tile_kernels.quant.norm_forward(
            x,
            weight,
            eps,
            residual=unfused_residual,
            residual_out=unfused_residual,
        )
        return tile_kernels.quant.per_token_cast(normalized, **cast_args)

    for _ in range(10):
        wrapper()
        direct()
        unfused()
    torch.cuda.synchronize()

    wrapper_out = wrapper()
    wrapper_us = benchmark_timer(wrapper, warmup=200, rep=100)
    direct_us = benchmark_timer(direct, warmup=200, rep=100)
    unfused_us = benchmark_timer(unfused, warmup=200, rep=100)
    params = {'num_tokens': num_tokens, 'hidden': hidden, 'num_per_channels': 128}
    bandwidth_gbs = count_bytes(x, wrapper_residual, weight, wrapper_out) / wrapper_us / 1e3
    benchmark_record(
        kernel='add_rmsnorm_forward_and_per_token_cast',
        operation='fwd',
        params=params,
        time_us=wrapper_us,
        bandwidth_gbs=bandwidth_gbs,
        extras={'direct_us': direct_us, 'unfused_us': unfused_us},
    )
    benchmark_record(
        kernel='norm_forward_and_per_token_cast_alias',
        operation='fwd',
        params=params,
        time_us=direct_us,
    )
    benchmark_record(
        kernel='unfused_norm_forward_per_token_cast',
        operation='fwd',
        params=params,
        time_us=unfused_us,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0), ids=make_param_id)
def test_norm_forward_and_per_token_cast_benchmark(benchmark_timer, benchmark_record, params):
    fmt = params['fmt']
    eps = params['eps']
    num_per_channels = params['num_per_channels']
    with_out_bf16 = params['with_out_bf16']
    mega_moe_block_m = params['mega_moe_block_m']

    x, weight, residual, mega_moe_sf = generate_test_data(params)
    args = dict(x=x, weight=weight, eps=eps, residual=residual, out_scale=params['out_scale'])
    if fmt == 'e4m3':
        func = lambda: tile_kernels.quant.norm_forward_and_per_token_cast(
            **args,
            fmt=fmt,
            num_per_channels=num_per_channels,
            **get_cast_params(params),
            with_out_bf16=with_out_bf16,
            mega_moe_sf=mega_moe_sf[: x.shape[0]] if mega_moe_sf is not None else None,
            mega_moe_block_m=mega_moe_block_m,
        )
    else:
        func = lambda: tile_kernels.quant.norm_forward(**args)
    result = func()
    t_us = benchmark_timer(func)
    # Without residual, norm_input aliases x and does not incur another write.
    num_bytes = count_bytes(
        x,
        weight,
        residual,
        *result[:-1],
        result[-1] if residual is not None else None,
        mega_moe_sf[: x.shape[0]] if mega_moe_sf is not None else None,
    )
    benchmark_record(
        kernel='norm_forward_and_per_token_cast',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )

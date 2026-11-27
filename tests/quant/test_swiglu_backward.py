import os
import pytest
import torch

import tile_kernels
from tile_kernels.rand import randn
from tile_kernels.utils import str_to_dtype
from tile_kernels.config import set_token_alignment, get_device, is_ascend
from tile_kernels.testing.generator import (
    generate_samples,
    generate_psum_layout,
    generate_hidden_sizes,
    generate_moe_params,
    generate_cast_config,
    get_test_level,
)
from tile_kernels.testing.numeric import assert_equal, count_bytes
from tile_kernels.testing.bench import make_param_id

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_test_data(params):
    hidden = params['hidden']
    num_per_channels = params['num_per_channels']
    with_weights = params['with_weights']
    fmt = params['fmt']
    with_psum_layout = params['with_psum_layout']

    mask, psum_num_tokens_per_expert = generate_psum_layout(params)
    num_expanded_tokens = mask.numel()
    psum_num_tokens_per_expert = psum_num_tokens_per_expert if with_psum_layout else None

    x_dtype = torch.bfloat16 if fmt == 'e4m3' else str_to_dtype(fmt)
    x = randn((num_expanded_tokens, hidden * 2), dtype=x_dtype, device=get_device())
    x[~mask] = 0
    if fmt == 'e4m3':
        x = tile_kernels.quant.per_token_cast(x, 'e4m3', num_per_channels=num_per_channels)
    weighted_act_x_grad = randn((num_expanded_tokens, hidden), dtype=x_dtype, device=get_device())
    weighted_act_x_grad[~mask] = 0

    topk_weights = None
    if with_weights:
        topk_weights = torch.rand((num_expanded_tokens,), dtype=torch.float32, device=get_device())
        topk_weights[~mask] = 0

    return x, topk_weights, psum_num_tokens_per_expert, weighted_act_x_grad


def extract(values, fmt, with_weights, has_act_out):
    values = list(values)
    x_grad = values.pop(0)
    x_grad_fp8 = values.pop(0) if fmt == 'e4m3' else None
    weight_grad = values.pop(0) if with_weights else None
    act_out = values.pop(0) if has_act_out else None
    assert not values
    return x_grad, x_grad_fp8, weight_grad, act_out


def generate_test_params(level: int) -> list[dict]:
    params = [
        {
            **moe,
            **args,
            'hidden': hidden_size,
            'with_psum_layout': with_psum_layout,
            'num_per_channels': num_per_channels,
            'fmt': fmt,
            'use_tma_aligned_col_major_sf': use_tma_aligned_col_major_sf,
            'round_sf': round_sf,
            'use_packed_ue8m0': use_packed_ue8m0,
            'routed_scaling_factor': routed_scaling_factor,
        }
        for moe in generate_moe_params(level)
        for fmt in ('e4m3', 'bf16', 'fp32')
        for num_per_channels in ((32,) if fmt == 'e4m3' else (None,))
        for hidden_size in generate_hidden_sizes(level, ['ffn'], align=128 if is_ascend() else max(64, num_per_channels or 1))
        for use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0 in generate_cast_config(level, fmt)
        for with_psum_layout in ((True,) if level == 0 else (True, False))
        for args in generate_samples(
            level,
            clamp_value=(None, 0.5),
            do_recompute=(True, False),
            with_weights=(True, False),
            alignment=(256 if with_psum_layout or fmt == 'e4m3' else 1,),
        )
        for routed_scaling_factor in (((2.5,) if level == 0 else (2.5, None)) if args['with_weights'] else (None,))
    ]
    return params


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_swiglu_backward_and_per_token_cast(params):
    num_per_channels = params['num_per_channels']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']
    round_sf = params['round_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']
    clamp_value = params['clamp_value']
    with_weights = params['with_weights']
    fmt = params['fmt']
    alignment = params['alignment']
    do_recompute = params['do_recompute']
    routed_scaling_factor = params['routed_scaling_factor']

    set_token_alignment(alignment)
    x, topk_weights, psum_num_tokens_per_expert, weighted_act_x_grad = generate_test_data(params)

    if fmt == 'e4m3':
        func = lambda: tile_kernels.quant.swiglu_backward_and_per_token_cast(
            x,
            weighted_act_x_grad,
            'e4m3',
            num_per_channels,
            psum_num_tokens_per_expert,
            topk_weights,
            routed_scaling_factor=routed_scaling_factor,
            use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
            round_sf=round_sf,
            use_packed_ue8m0=use_packed_ue8m0,
            clamp_value=clamp_value,
            do_recompute=do_recompute,
        )
    else:
        func = lambda: tile_kernels.quant.swiglu_backward(
            x,
            weighted_act_x_grad,
            fmt,
            psum_num_tokens_per_expert,
            topk_weights,
            routed_scaling_factor=routed_scaling_factor,
            clamp_value=clamp_value,
            do_recompute=do_recompute,
        )

    def func_ref():
        if fmt == 'e4m3':
            x_fp32 = tile_kernels.quant.cast_back(x, 'fp32', x_block_size=(1, num_per_channels))
        else:
            x_fp32 = x.float()

        result = list(
            tile_kernels.torch.swiglu_backward(
                x_fp32,
                weighted_act_x_grad,
                psum_num_tokens_per_expert=psum_num_tokens_per_expert,
                topk_weights=topk_weights,
                routed_scaling_factor=routed_scaling_factor,
                clamp_value=clamp_value,
                alignment=alignment,
            )
        )
        x_grad_full, _, weight_grad, out = extract(result, 'bf16', with_weights, True)

        # Build return in the new field order
        out_dtype = torch.bfloat16 if fmt != 'fp32' else torch.float32
        result = [x_grad_full.to(out_dtype)]
        if fmt == 'e4m3':
            x_grad_fp8 = tile_kernels.torch.cast(
                x_grad_full,
                'e4m3',
                block_size=(1, num_per_channels),
                round_sf=round_sf,
                use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
                use_packed_ue8m0=use_packed_ue8m0,
            )
            result.append(x_grad_fp8)
        if with_weights:
            result.append(weight_grad)
        if do_recompute:
            result.append(out.to(out_dtype))
        return tuple(result)

    x_grad, x_grad_fp8, topk_weights_grad, weighted_act_x = extract(func(), fmt, with_weights, has_act_out=do_recompute)
    x_grad_ref, x_grad_fp8_ref, topk_weights_grad_ref, weighted_act_x_ref = extract(func_ref(), fmt, with_weights, has_act_out=do_recompute)

    assert_equal(x_grad, x_grad_ref)
    if fmt == 'e4m3':
        x_grad_dequant = tile_kernels.torch.cast_back(x_grad_fp8, 'fp32', block_size=(1, num_per_channels))
        x_grad_dequant_ref = tile_kernels.torch.cast_back(x_grad_fp8_ref, 'fp32', block_size=(1, num_per_channels))
        assert_equal(x_grad_dequant, x_grad_dequant_ref)
    if with_weights:
        torch.testing.assert_close(topk_weights_grad, topk_weights_grad_ref, atol=1e-4, rtol=1e-4)
    if do_recompute:
        assert_equal(weighted_act_x, weighted_act_x_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0), ids=make_param_id)
def test_swiglu_backward_and_per_token_cast_benchmark(benchmark_timer, benchmark_record, params):
    num_per_channels = params['num_per_channels']
    round_sf = params['round_sf']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']
    clamp_value = params['clamp_value']
    fmt = params['fmt']
    alignment = params['alignment']
    do_recompute = params['do_recompute']
    routed_scaling_factor = params['routed_scaling_factor']

    set_token_alignment(alignment)
    x, topk_weights, psum_num_tokens_per_expert, weighted_act_x_grad = generate_test_data(params)

    if fmt == 'e4m3':
        func = lambda: tile_kernels.quant.swiglu_backward_and_per_token_cast(
            x,
            weighted_act_x_grad,
            'e4m3',
            num_per_channels,
            psum_num_tokens_per_expert,
            topk_weights,
            routed_scaling_factor=routed_scaling_factor,
            use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
            round_sf=round_sf,
            use_packed_ue8m0=use_packed_ue8m0,
            clamp_value=clamp_value,
            do_recompute=do_recompute,
        )
    else:
        func = lambda: tile_kernels.quant.swiglu_backward(
            x,
            weighted_act_x_grad,
            fmt,
            psum_num_tokens_per_expert,
            topk_weights,
            routed_scaling_factor=routed_scaling_factor,
            clamp_value=clamp_value,
            do_recompute=do_recompute,
        )

    result_list = func()
    t_us = benchmark_timer(func)
    num_bytes = count_bytes(
        x,
        weighted_act_x_grad,
        topk_weights,
        *result_list,
    )

    benchmark_record(
        kernel='swiglu_backward_and_per_token_cast',
        operation='bwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )

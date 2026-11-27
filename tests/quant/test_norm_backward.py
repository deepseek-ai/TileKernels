import math
import os
import pytest
import torch

import tile_kernels
from tile_kernels.config import get_device, get_device_num_sms, is_ascend
from tile_kernels.rand import randn
from tile_kernels.testing.bench import make_param_id
from tile_kernels.testing.generator import (
    generate_samples,
    generate_num_tokens,
    generate_hidden_sizes,
    generate_num_sms,
    get_test_level,
)
from tile_kernels.testing.numeric import count_bytes
from tile_kernels.torch import norm_forward_ref, norm_backward_ref
from tile_kernels.utils import str_to_dtype

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


_LEVEL0_NORM_CASES = [
    (16384, 5120, True, False, 1.0),
]


def generate_test_data(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    fmt = params['fmt']
    dtype = str_to_dtype(fmt)
    out_grad_dtype = str_to_dtype(params['out_grad_fmt'] or fmt)
    weight_dtype = str_to_dtype(params['weight_fmt'] or fmt)
    eps = params['eps']
    x = randn((num_tokens, hidden + params['row_padding']), dtype=dtype, device=get_device())[:, :hidden]
    weight = randn((hidden,), dtype=weight_dtype, device=x.device) if params['use_weight'] else None
    _, rstd, norm_input = norm_forward_ref(x, weight, eps, out_scale=params['out_scale'], rms_group_size=params['rms_group_size'])
    out_grad = randn(x.shape, dtype=out_grad_dtype, device=x.device)
    residual_out_grad = randn(x.shape, dtype=dtype, device=x.device) if params['has_residual_out_grad'] else None
    weight_grad = (
        randn(weight.shape, dtype=str_to_dtype(params['weight_grad_fmt']), device=x.device)
        if weight is not None and params['accumulate'] and params['with_weight_grad']
        else None
    )
    return norm_input, weight, rstd, out_grad, residual_out_grad, weight_grad


def generate_test_params(level: int) -> list[dict]:
    hidden_sizes = generate_hidden_sizes(level, ['main', 'q_lora', 'head', 'group', 'vision', 'engram_input', 'wv'], align=128 if is_ascend() else 32)
    dtype_combos = [
        (fmt, out_grad_fmt, weight_fmt, x_grad_fmt)
        for fmt in ('bf16', 'fp32')
        for out_grad_fmt in ('bf16', 'fp32')
        for weight_fmt in ('bf16', 'fp32')
        for x_grad_fmt in ('bf16', 'fp32')
    ]
    params = [
        {
            **args,
            'hidden': hidden,
            'num_tokens': num_tokens,
            'fmt': fmt,
            'out_grad_fmt': out_grad_fmt,
            'weight_fmt': weight_fmt,
            'x_grad_fmt': x_grad_fmt,
            'use_weight': use_weight,
            'weight_grad_fmt': weight_grad_fmt,
            'eps': eps,
        }
        for hidden in hidden_sizes
        for num_tokens in generate_num_tokens(level, backward_only=True)
        for eps in (1e-6,)
        for fmt, out_grad_fmt, weight_fmt, x_grad_fmt in dtype_combos
        for use_weight in (True, False)
        if use_weight or weight_fmt != 'fp32'
        for args in generate_samples(
            level,
            rms_group_size=(None,) if level == 0 else (None, 128, 1024),
            has_out_grad=(True,),
            has_residual_out_grad=(False, True),
            accumulate=(False, True) if use_weight else (False,),
            out_scale=(1.0, 0.3),
            row_padding=(0, 16),
            num_sms=generate_num_sms(level),
            do_recompute=(False,) if level == 0 else (False, True),
            with_weight_grad=(use_weight,) if level == 0 else ((True, False) if use_weight else (False,)),
        )
        if hidden <= 12288 or args['rms_group_size'] is not None
        if args['rms_group_size'] is None or hidden % args['rms_group_size'] == 0
        # Without accumulation, weight_grad has the weight dtype.
        for weight_grad_fmt in (('bf16', 'fp32') if args['accumulate'] and args['with_weight_grad'] else (None,))
    ]
    if level == 0:
        params.extend(
            {
                'num_tokens': num_tokens,
                'hidden': hidden,
                'fmt': 'bf16',
                'out_grad_fmt': None,
                'weight_fmt': None,
                'x_grad_fmt': None,
                'use_weight': use_weight,
                'weight_grad_fmt': 'fp32' if use_weight else None,
                'eps': 1e-6 if has_residual_out_grad else 1e-20,
                'has_out_grad': True,
                'has_residual_out_grad': has_residual_out_grad,
                'accumulate': use_weight,
                'out_scale': out_scale,
                'row_padding': 0,
                'num_sms': get_device_num_sms(),
                'rms_group_size': None,
                'do_recompute': False,
                'with_weight_grad': use_weight,
            }
            for num_tokens, hidden, use_weight, has_residual_out_grad, out_scale in _LEVEL0_NORM_CASES
        )
    return params


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_norm_backward(params):
    out_scale = params['out_scale']
    x_grad_dtype = str_to_dtype(params['x_grad_fmt']) if params['x_grad_fmt'] else None

    tile_kernels.set_num_sms(params['num_sms'])
    norm_input, weight, rstd, out_grad, residual_out_grad, weight_grad = generate_test_data(params)
    out = torch.empty_like(norm_input) if params['do_recompute'] else None
    func = lambda: tile_kernels.quant.norm_backward(
        out_grad=out_grad,
        x=norm_input,
        weight=weight,
        rstd=rstd,
        rms_group_size=params['rms_group_size'],
        with_weight_grad=params['with_weight_grad'],
        out=out,
        residual_out_grad=residual_out_grad,
        out_scale=out_scale,
        weight_grad=weight_grad,
        x_grad_dtype=x_grad_dtype,
    )

    def func_ref():
        x_grad, weight_grad_ref = norm_backward_ref(
            out_grad,
            norm_input,
            weight,
            rstd,
            residual_out_grad,
            out_scale,
            x_grad_dtype=x_grad_dtype,
            rms_group_size=params['rms_group_size'],
        )
        if weight_grad is not None:
            weight_grad_ref = (weight_grad.float() + weight_grad_ref).to(weight_grad.dtype)
        elif weight_grad_ref is not None:
            weight_grad_ref = weight_grad_ref.to(weight.dtype)
        return x_grad, weight_grad_ref

    x_grad_ref, weight_grad_ref = func_ref()
    x_grad, weight_grad_out = func()
    if out is not None:
        out_ref, _, _ = norm_forward_ref(norm_input, weight, params['eps'], out_scale=out_scale, rms_group_size=params['rms_group_size'])
        torch.testing.assert_close(out, out_ref)
    assert x_grad.is_contiguous() and x_grad.dtype == (x_grad_dtype or norm_input.dtype)
    torch.testing.assert_close(x_grad, x_grad_ref)
    if weight is not None and params['with_weight_grad']:
        atol = 1e-5 * math.sqrt(params['num_tokens'] / params['num_sms'])
        rtol = 1e-4 if weight_grad_out.dtype == torch.float32 else 8e-3
        torch.testing.assert_close(weight_grad_out, weight_grad_ref, atol=atol, rtol=rtol)
    else:
        assert weight_grad_out is None
    if weight_grad is not None:
        assert weight_grad_out is weight_grad


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0), ids=make_param_id)
def test_norm_backward_benchmark(benchmark_timer, benchmark_record, params):
    out_scale = params['out_scale']

    tile_kernels.set_num_sms(params['num_sms'])
    norm_input, weight, rstd, out_grad, residual_out_grad, weight_grad = generate_test_data(params)
    out = torch.empty_like(norm_input) if params['do_recompute'] else None
    func = lambda: tile_kernels.quant.norm_backward(
        out_grad=out_grad,
        x=norm_input,
        weight=weight,
        rstd=rstd,
        rms_group_size=params['rms_group_size'],
        with_weight_grad=params['with_weight_grad'],
        out=out,
        residual_out_grad=residual_out_grad,
        out_scale=out_scale,
        weight_grad=weight_grad,
        x_grad_dtype=str_to_dtype(params['x_grad_fmt']) if params['x_grad_fmt'] else None,
    )
    result = func()
    t_us = benchmark_timer(func)
    num_bytes = count_bytes(out_grad, norm_input, weight, rstd, residual_out_grad, weight_grad, *result, out)
    benchmark_record(
        kernel='norm_backward',
        operation='bwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )

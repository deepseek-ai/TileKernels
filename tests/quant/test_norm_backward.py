import math
import os
import pytest
import torch

import tile_kernels
from tile_kernels.config import get_device, is_ascend
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


def generate_test_data(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    dtype = str_to_dtype(params['fmt'])
    eps = params['eps']
    x = randn((num_tokens, hidden + params['row_padding']), dtype=dtype, device=get_device())[:, :hidden]
    weight = randn((hidden,), dtype=dtype, device=x.device) if params['use_weight'] else None
    _, rstd, norm_input = norm_forward_ref(x, weight, eps, out_scale=params['out_scale'])
    out_grad = randn(x.shape, dtype=dtype, device=x.device) if params['has_out_grad'] else torch.zeros_like(x)
    residual_out_grad = randn(x.shape, dtype=dtype, device=x.device) if params['has_residual_out_grad'] else None
    weight_grad = (
        randn(weight.shape, dtype=str_to_dtype(params['weight_grad_fmt']), device=x.device) if weight is not None and params['accumulate'] else None
    )
    return norm_input, weight, rstd, out_grad, residual_out_grad, weight_grad


def generate_test_params(level: int) -> list[dict]:
    params = [
        {
            **args,
            'hidden': hidden,
            'num_tokens': num_tokens,
            'fmt': fmt,
            'use_weight': use_weight,
            'weight_grad_fmt': weight_grad_fmt,
            'eps': eps,
        }
        for hidden in ([4096, 5120, 7168] if level == 0 else generate_hidden_sizes(128 if is_ascend() else 32))
        for num_tokens in generate_num_tokens(level, backward_only=True)
        for eps in (1e-6,)
        for fmt in ('bf16', 'fp32')
        for use_weight in (True, False)
        for args in generate_samples(
            level,
            has_out_grad=(True,),
            has_residual_out_grad=(False, True),
            accumulate=(False, True) if use_weight else (False,),
            out_scale=(1.0, 0.3),
            row_padding=(0, 16),
            num_sms=generate_num_sms(level),
        )
        # weight_grad_fmt only matters when accumulating; otherwise it always equals fmt.
        for weight_grad_fmt in (('bf16', 'fp32') if args['accumulate'] else (None,))
    ]
    return params


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_norm_backward(params):
    out_scale = params['out_scale']

    tile_kernels.set_num_sms(params['num_sms'])
    norm_input, weight, rstd, out_grad, residual_out_grad, weight_grad = generate_test_data(params)
    func = lambda: tile_kernels.quant.norm_backward(
        out_grad=out_grad,
        x=norm_input,
        weight=weight,
        rstd=rstd,
        residual_out_grad=residual_out_grad,
        out_scale=out_scale,
        weight_grad=weight_grad,
    )

    def func_ref():
        x_grad, weight_grad_ref = norm_backward_ref(out_grad, norm_input, weight, rstd, residual_out_grad, out_scale)
        if weight_grad is not None:
            weight_grad_ref = (weight_grad.float() + weight_grad_ref).to(weight_grad.dtype)
        elif weight_grad_ref is not None:
            weight_grad_ref = weight_grad_ref.to(weight.dtype)
        return x_grad, weight_grad_ref

    x_grad_ref, weight_grad_ref = func_ref()
    x_grad, weight_grad_out = func()
    assert x_grad.is_contiguous()
    torch.testing.assert_close(x_grad, x_grad_ref)
    if weight is not None:
        # dW sums num_tokens products in a serial fp32 chain of length num_tokens / num_sms, hence the 1 / sqrt(num_sms) term.
        atol = 2e-3 / math.sqrt(params['num_sms'])
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
    func = lambda: tile_kernels.quant.norm_backward(
        out_grad=out_grad,
        x=norm_input,
        weight=weight,
        rstd=rstd,
        residual_out_grad=residual_out_grad,
        out_scale=out_scale,
        weight_grad=weight_grad,
    )
    result = func()
    t_us = benchmark_timer(func)
    num_bytes = count_bytes(out_grad, norm_input, weight, rstd, residual_out_grad, weight_grad, *result)
    benchmark_record(
        kernel='norm_backward',
        operation='bwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )

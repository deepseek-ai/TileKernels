import os

import pytest
import torch

from tile_kernels.config import get_device
from tile_kernels.engram import engram_gate_fwd
from tile_kernels.rand import randn
from tile_kernels.testing.bench import make_param_id
from tile_kernels.testing.generator import generate_hidden_sizes, generate_num_tokens, get_test_level
from tile_kernels.testing.numeric import assert_equal, count_bytes
from tile_kernels.torch.engram import engram_gate_ref
from tile_kernels.utils import str_to_dtype

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_test_data(params):
    num_tokens = params['num_tokens']
    hc_mult = params['hc']
    hidden_size = params['hidden']
    dtype = str_to_dtype(params['dtype'])
    wv_dtype = str_to_dtype(params['wv_dtype'])
    has_value_norm = params['has_value_norm']
    eps = 1e-20
    clamp_value = 1e-6
    device = get_device()
    x_data = randn(num_tokens, hc_mult, hidden_size, dtype=dtype, device=device)
    kv_data = randn(num_tokens, hc_mult + 1, hidden_size, dtype=torch.bfloat16, device=device)
    wh_data = randn(hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    we_data = randn(hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    weight_xk_fused = wh_data.float() * we_data.float()
    weight_value = randn(hidden_size, dtype=wv_dtype, device=device) if has_value_norm else None
    image_token_mask = (torch.arange(num_tokens, device=device) % 5 < 2) if params['with_mask'] else None
    return (x_data, kv_data, wh_data, we_data, weight_xk_fused, eps, clamp_value, image_token_mask, weight_value)


def generate_test_params(level: int, mask_options: tuple[bool, ...] = (False, True)) -> list[dict]:
    return [
        {
            'num_tokens': t,
            'hc': hc,
            'hidden': hidden_size,
            'with_mask': with_mask,
            'dtype': dtype,
            'wv_dtype': wv_dtype,
            'has_value_norm': has_value_norm,
        }
        for t in generate_num_tokens(level)
        for hc in (4,)
        for hidden_size in generate_hidden_sizes(level, ['main'], align=128)
        for with_mask in mask_options
        for dtype in ('bf16', 'fp32')
        for has_value_norm in (False, True)
        for wv_dtype in (['bf16', 'fp32'] if has_value_norm else ['bf16'])
    ]


@pytest.mark.parametrize('params', generate_test_params(max(get_test_level(), 1)), ids=make_param_id)
def test_engram_gate_fwd(params):
    x_data, kv_data, wh_data, we_data, weight_xk_fused, eps, clamp_value, image_token_mask, weight_value = generate_test_data(params)
    out_ref, dot_ref, gate_score_ref, rstd_x_ref, rstd_kv_ref = engram_gate_ref(
        x_data,
        kv_data,
        wh_data,
        we_data,
        weight_value,
        clamp_value,
        eps,
        save_for_backward=True,
        image_token_mask=image_token_mask,
    )

    # Correctness: save_for_backward=True
    out_save, dot, gate_score, rstd_x, rstd_kv = engram_gate_fwd(
        x_data,
        kv_data,
        weight_xk_fused,
        weight_value,
        eps,
        clamp_value,
        save_for_backward=True,
        image_token_mask=image_token_mask,
    )
    assert dot is not None and gate_score is not None and rstd_x is not None and rstd_kv is not None
    text_tokens = slice(None) if image_token_mask is None else ~image_token_mask
    torch.testing.assert_close(out_save, out_ref, rtol=7e-3, atol=7e-3)
    torch.testing.assert_close(dot[text_tokens], dot_ref[text_tokens], rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(gate_score[text_tokens], gate_score_ref[text_tokens], rtol=4e-4, atol=4e-4)
    torch.testing.assert_close(rstd_x[text_tokens], rstd_x_ref[text_tokens])
    torch.testing.assert_close(rstd_kv[text_tokens], rstd_kv_ref[text_tokens])

    if image_token_mask is not None:
        assert_equal(out_save[image_token_mask], x_data[image_token_mask])

    # Correctness: save_for_backward=False
    out_no_save, dot_n, gate_score_n, rstd_x_n, rstd_kv_n = engram_gate_fwd(
        x_data,
        kv_data,
        weight_xk_fused,
        weight_value,
        eps,
        clamp_value,
        save_for_backward=False,
        image_token_mask=image_token_mask,
    )
    assert dot_n is None and gate_score_n is None and rstd_x_n is None and rstd_kv_n is None
    assert_equal(out_no_save, out_save)

    # Correctness: inplace=True overwrites hidden_states with the same result
    x_inplace = x_data.clone()
    out_inplace = engram_gate_fwd(
        x_inplace,
        kv_data,
        weight_xk_fused,
        weight_value,
        eps,
        clamp_value,
        save_for_backward=False,
        image_token_mask=image_token_mask,
        inplace=True,
    )[0]
    assert_equal(out_inplace, out_save)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'params',
    [{**params, 'save': save} for params in generate_test_params(1, mask_options=(False,)) for save in (False, True)],
    ids=make_param_id,
)
def test_engram_gate_fwd_benchmark(benchmark_timer, benchmark_record, params):
    (x_data, kv_data, _, _, weight_xk_fused, eps, clamp_value, _, weight_value) = generate_test_data(params)
    save_for_backward = params['save']

    out, dot, gate_score, rstd_x, rstd_kv = engram_gate_fwd(
        x_data,
        kv_data,
        weight_xk_fused,
        weight_value,
        eps,
        clamp_value,
        save_for_backward=save_for_backward,
    )
    t_us = benchmark_timer(
        lambda: engram_gate_fwd(
            x_data,
            kv_data,
            weight_xk_fused,
            weight_value,
            eps,
            clamp_value,
            save_for_backward=save_for_backward,
        )
    )
    num_bytes = count_bytes(x_data, kv_data, weight_xk_fused, weight_value, out, dot, gate_score, rstd_x, rstd_kv)

    benchmark_record(
        kernel='engram_gate_fwd',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )

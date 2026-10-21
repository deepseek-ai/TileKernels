import os

import pytest
import torch

from tile_kernels.config import get_device, is_ascend
from tile_kernels.engram import engram_gate_fwd
from tile_kernels.rand import randn
from tile_kernels.testing.bench import make_param_id
from tile_kernels.testing.generator import generate_hidden_sizes, generate_num_tokens, get_test_level
from tile_kernels.testing.numeric import assert_equal, count_bytes
from tile_kernels.torch.engram import engram_gate_ref

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_test_data(params):
    num_tokens = params['num_tokens']
    hc_mult = params['hc']
    hidden_size = params['hidden']
    eps = 1e-20
    clamp_value = 1e-6
    device = get_device()
    x_data = randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    kv_data = randn(num_tokens, hc_mult + 1, hidden_size, dtype=torch.bfloat16, device=device)
    wh_data = randn(hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    we_data = randn(hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    weight_fused = wh_data.float() * we_data.float()
    image_token_mask = (torch.arange(num_tokens, device=device) % 5 < 2) if params['with_mask'] else None
    return (x_data, kv_data, wh_data, we_data, weight_fused, eps, clamp_value, image_token_mask)


def generate_test_params(level: int, mask_options: tuple[bool, ...] = (False, True)) -> list[dict]:
    return [
        {'num_tokens': t, 'hc': hc, 'hidden': hidden_size, 'with_mask': with_mask}
        for t in generate_num_tokens(level)
        for hc in (4,)
        for hidden_size in generate_hidden_sizes(128 if is_ascend() else 256)
        for with_mask in mask_options
    ]


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_engram_gate_fwd(params):
    x_data, kv_data, wh_data, we_data, weight_fused, eps, clamp_value, image_token_mask = generate_test_data(params)
    out_ref, dot_ref, gate_score_ref, rstd_x_ref, rstd_k_ref = engram_gate_ref(
        x_data,
        kv_data,
        wh_data,
        we_data,
        clamp_value,
        eps,
        save_for_backward=True,
        image_token_mask=image_token_mask,
    )

    # Correctness: save_for_backward=True
    out_save, dot, gate_score, rstd_x, rstd_k = engram_gate_fwd(
        x_data,
        kv_data,
        weight_fused,
        eps,
        clamp_value,
        save_for_backward=True,
        image_token_mask=image_token_mask,
    )
    assert dot is not None and gate_score is not None and rstd_x is not None and rstd_k is not None
    text_tokens = slice(None) if image_token_mask is None else ~image_token_mask
    torch.testing.assert_close(out_save, out_ref, rtol=8e-3, atol=8e-3)
    torch.testing.assert_close(dot[text_tokens], dot_ref[text_tokens], rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(gate_score[text_tokens], gate_score_ref[text_tokens], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(rstd_x[text_tokens], rstd_x_ref[text_tokens])
    torch.testing.assert_close(rstd_k[text_tokens], rstd_k_ref[text_tokens])

    if image_token_mask is not None:
        assert_equal(out_save[image_token_mask], x_data[image_token_mask])

    # Correctness: save_for_backward=False
    out_no_save, dot_n, gate_score_n, rstd_x_n, rstd_k_n = engram_gate_fwd(
        x_data,
        kv_data,
        weight_fused,
        eps,
        clamp_value,
        save_for_backward=False,
        image_token_mask=image_token_mask,
    )
    assert dot_n is None and gate_score_n is None and rstd_x_n is None and rstd_k_n is None
    assert_equal(out_no_save, out_save)

    # Correctness: inplace=True overwrites hidden_states with the same result
    x_inplace = x_data.clone()
    out_inplace = engram_gate_fwd(
        x_inplace,
        kv_data,
        weight_fused,
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
    [{**params, 'save': save} for params in generate_test_params(0, mask_options=(False,)) for save in (False, True)],
    ids=make_param_id,
)
def test_engram_gate_fwd_benchmark(benchmark_timer, benchmark_record, params):
    (x_data, kv_data, _, _, weight_fused, eps, clamp_value, _) = generate_test_data(params)
    save_for_backward = params['save']

    out, dot, gate_score, rstd_x, rstd_k = engram_gate_fwd(
        x_data,
        kv_data,
        weight_fused,
        eps,
        clamp_value,
        save_for_backward=save_for_backward,
    )
    t_us = benchmark_timer(
        lambda: engram_gate_fwd(
            x_data,
            kv_data,
            weight_fused,
            eps,
            clamp_value,
            save_for_backward=save_for_backward,
        )
    )
    num_bytes = count_bytes(x_data, kv_data, weight_fused, out, dot, gate_score, rstd_x, rstd_k)

    benchmark_record(
        kernel='engram_gate_fwd',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )

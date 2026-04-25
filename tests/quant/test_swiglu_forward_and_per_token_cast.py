import os
import pytest
import torch

import tile_kernels
from tile_kernels.testing import clear_unused_sf
from tile_kernels.torch import swiglu_forward, cast
from tile_kernels.config import set_num_sms
from tile_kernels.testing.generator import generate_topk_idx, generate_hidden_sizes, generate_moe_params, generate_num_sms
from tile_kernels.testing.numeric import assert_equal, count_bytes
from tile_kernels.testing.bench import make_param_id
from tests.conftest import IS_HIP

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'

# swiglu_forward depends on get_fused_mapping which uses __match_any_sync (no AMD equivalent)
pytestmark = pytest.mark.skipif(IS_HIP, reason='swiglu_forward_and_per_token_cast depends on get_fused_mapping which uses __match_any_sync (no HIP/AMD equivalent)')



def generate_test_data(params):
    num_topk = params['num_topk']
    num_experts = params['num_experts']
    hidden = params['hidden']

    topk_idx = generate_topk_idx(params)
    num_tokens = topk_idx.shape[0]
    alignment = 16
    pos_to_expert, pos_to_token, pos_to_token_topk, token_topk_to_pos, _, _, _, _ = tile_kernels.moe.get_fused_mapping(topk_idx, num_experts, 0, alignment)
    post_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')
    x = torch.randn((num_tokens, hidden * 2), dtype=torch.bfloat16, device='cuda')
    x = tile_kernels.moe.expand_to_fused(x, token_topk_to_pos, pos_to_token)
    _clamped_count = torch.zeros(3, dtype=torch.int64, device='cuda')

    return (x, num_tokens, pos_to_expert, pos_to_token_topk, post_weights, _clamped_count)


def generate_test_params(is_benchmark: bool) -> list[dict]:
    params = [
        {
            **moe,
            'hidden': hidden_size,
            'enable_pos_to_expert': enable_pos_to_expert,
            'with_weights': with_weights,
            'num_per_channels': num_per_channels,
            'use_tma_aligned_col_major_sf': use_tma_aligned_col_major_sf,
            'round_sf': round_sf,
            'use_packed_ue8m0': use_packed_ue8m0,
            'swiglu_clamp_value': swiglu_clamp_value,
        }
        for moe in generate_moe_params(is_benchmark)
        for hidden_size in [h // 2 for h in generate_hidden_sizes(256)]
        for enable_pos_to_expert in (True, False)
        for with_weights in (True, False)
        for num_per_channels in (128, hidden_size)
        for use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0 in (
            [(False, True, False)] if IS_HIP else [(False, True, False), (True, True, True)]
        )
        if not ((use_packed_ue8m0 and with_weights) or (use_tma_aligned_col_major_sf and num_per_channels == hidden_size))
        for swiglu_clamp_value in (None, 10.0, 0.5)
    ]
    if is_benchmark:
        params = [p for p in params if p['swiglu_clamp_value'] in (0.5, None) and p['use_tma_aligned_col_major_sf'] and p['round_sf']]
    return params


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_swiglu_forward_and_per_token_cast(params):
    hidden = params['hidden']
    enable_pos_to_expert = params['enable_pos_to_expert']
    with_weights = params['with_weights']
    num_per_channels = params['num_per_channels']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']
    round_sf = params['round_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']
    swiglu_clamp_value = params['swiglu_clamp_value']

    x, num_tokens, pos_to_expert, pos_to_token_topk, post_weights, _clamped_count = generate_test_data(params)

    do_clamp_count = num_per_channels != hidden and swiglu_clamp_value is not None
    clamped_count_ref = _clamped_count.clone() if do_clamp_count else None
    base_args = dict(
        x=x,
        fmt='e4m3',
        pos_to_expert=pos_to_expert if enable_pos_to_expert else None,
        swiglu_clamp_value=swiglu_clamp_value
    )
    kernel_args = dict(
        **base_args,
        num_per_channels=num_per_channels,
        pos_to_token_topk=pos_to_token_topk if with_weights else None,
        topk_weights=post_weights if with_weights else None,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf, round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )

    def func_ref():
        out = swiglu_forward(
            x,
            pos_to_token_topk if with_weights else None,
            post_weights if with_weights else None,
            swiglu_clamp_value,
            clamped_count_ref,
        )

        if enable_pos_to_expert:
            mask = pos_to_expert == -1
            out = out.masked_fill(mask.unsqueeze(-1), 0)

        result = cast(
            out, 'e4m3', (1, num_per_channels),
            use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
            round_sf=round_sf,
            use_packed_ue8m0=use_packed_ue8m0,
        )
        return result

    x_fp8_ref, x_sf_ref = func_ref()
    mask = (pos_to_expert == -1).unsqueeze(-1)
    x_fp8_ref_float = x_fp8_ref.float().masked_fill(mask, 0)
    x_sf_ref = x_sf_ref.masked_fill(mask, 0)
    if use_packed_ue8m0:
        x_sf_ref = clear_unused_sf(x_sf_ref, hidden, num_per_channels)

    for num_sms in generate_num_sms():
        set_num_sms(num_sms)
        clamped_count = _clamped_count.clone() if do_clamp_count else None
        x_fp8, x_sf = tile_kernels.quant.swiglu_forward_and_per_token_cast(
            **kernel_args, clamped_count=clamped_count
        )
        x_fp8_float = x_fp8.float().masked_fill(mask, 0)
        x_sf = x_sf.masked_fill(mask, 0)
        if use_packed_ue8m0:
            x_sf = clear_unused_sf(x_sf, hidden, num_per_channels)

        assert_equal(x_fp8_float, x_fp8_ref_float)
        assert_equal(x_sf, x_sf_ref)
        if do_clamp_count:
            assert_equal(clamped_count, clamped_count_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_swiglu_forward_and_per_token_cast_benchmark(benchmark_timer, benchmark_record, params):
    hidden = params['hidden']
    enable_pos_to_expert = params['enable_pos_to_expert']
    with_weights = params['with_weights']
    num_per_channels = params['num_per_channels']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']
    round_sf = params['round_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']
    swiglu_clamp_value = params['swiglu_clamp_value']

    x, num_tokens, pos_to_expert, pos_to_token_topk, post_weights, _clamped_count = generate_test_data(params)

    do_clamp_count = num_per_channels != hidden and swiglu_clamp_value is not None
    clamped_count = _clamped_count.clone() if do_clamp_count else None
    kernel_args = dict(
        x=x,
        fmt='e4m3',
        pos_to_expert=pos_to_expert if enable_pos_to_expert else None,
        swiglu_clamp_value=swiglu_clamp_value,
        num_per_channels=num_per_channels,
        pos_to_token_topk=pos_to_token_topk if with_weights else None,
        topk_weights=post_weights if with_weights else None,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf, round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )

    func = lambda: tile_kernels.quant.swiglu_forward_and_per_token_cast(
        **kernel_args, clamped_count=clamped_count
    )
    x_fp8, x_sf = func()

    t_us = benchmark_timer(func)
    num_bytes = count_bytes(x, x_fp8, x_sf)

    params.pop('num_send_tokens')
    benchmark_record(
        kernel='swiglu_forward_and_per_token_cast',
        operation='fwd',
        params={'num_tokens': num_tokens, **params},
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )

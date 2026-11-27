import pytest
import torch

from tile_kernels.config import get_device
from tile_kernels.mhc.multilayer_recompute_kernel import mhc_multilayer_recompute
from tile_kernels.mhc.post_kernel import mhc_post_fwd
from tile_kernels.mhc.pre_apply_mix_kernel import mhc_pre_apply_mix_fwd
from tile_kernels.rand import randn
from tile_kernels.testing.generator import generate_hidden_sizes, get_test_level

_CORRECTNESS_CASES = [
    (num_layers, num_post, hidden)
    for num_layers, num_post in [(1, 1), (3, 2), (3, 3), (10, 9), (10, 10)]
    for hidden in generate_hidden_sizes(max(get_test_level(), 1), ['main'], align=128)
]

_BENCH_CASES = [(10, num_post, 1, 8192, 4, h) for num_post in (9, 10) for h in generate_hidden_sizes(1, ['main'])]


def generate_multilayer_recompute_test_data(
    bs: int,
    seq: int,
    mhc_mult: int,
    hidden: int,
    num_layers: int,
    num_post: int,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
]:
    device = get_device()
    initial_residual = randn(bs, seq, mhc_mult, hidden, device=device, dtype=dtype)
    pre_mix_list = [randn(bs, seq, mhc_mult, 1, device=device, dtype=torch.float32) for _ in range(num_layers)]
    layer_output_list = [randn(bs, seq, hidden, device=device, dtype=dtype) for _ in range(num_post)]
    post_mix_list = [randn(bs, seq, mhc_mult, 1, device=device, dtype=torch.float32) for _ in range(num_post)]
    comb_mix_list = [randn(bs, seq, mhc_mult, mhc_mult, device=device, dtype=torch.float32) for _ in range(num_post)]
    layer_input_list = [torch.empty(bs, seq, hidden, device=device, dtype=dtype) for _ in range(num_layers)]
    residual_list = [torch.empty(bs, seq, mhc_mult, hidden, device=device, dtype=dtype) for _ in range(num_post)]
    return initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list, layer_input_list, residual_list


def _mhc_multilayer_recompute_ref(
    initial_residual: torch.Tensor,
    pre_mix_list: list[torch.Tensor],
    layer_output_list: list[torch.Tensor],
    post_mix_list: list[torch.Tensor],
    comb_mix_list: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    layer_input_refs: list[torch.Tensor] = []
    residual_refs: list[torch.Tensor] = []
    residual = initial_residual
    mhc_mult = initial_residual.shape[-2]
    hidden = initial_residual.shape[-1]
    num_tokens = initial_residual.numel() // (mhc_mult * hidden)
    for i in range(len(pre_mix_list)):
        layer_input = torch.empty(num_tokens, hidden, dtype=residual.dtype, device=residual.device)
        mhc_pre_apply_mix_fwd(
            residual.view(num_tokens, mhc_mult, hidden),
            pre_mix_list[i].view(num_tokens, mhc_mult),
            layer_input,
        )
        layer_input = layer_input.view(*residual.shape[:-2], hidden)
        layer_input_refs.append(layer_input)
        if i < len(layer_output_list):
            residual = mhc_post_fwd(layer_output_list[i], residual, post_mix_list[i], comb_mix_list[i])
            residual_refs.append(residual)
    return layer_input_refs, residual_refs


def _compute_io_bytes(n: int, mhc_mult: int, hidden: int, num_layers: int, num_post: int, element_size: int) -> tuple[int, int]:
    io_pre = n * mhc_mult * hidden * element_size + n * mhc_mult * 4 + n * hidden * element_size
    io_post = (
        n * hidden * element_size
        + n * mhc_mult * hidden * element_size
        + n * mhc_mult * 4
        + n * mhc_mult * mhc_mult * 4
        + n * mhc_mult * hidden * element_size
    )
    io_ref = num_layers * io_pre + num_post * io_post
    io_fused = (
        n * mhc_mult * hidden * element_size
        + num_layers * (n * mhc_mult * 4 + n * hidden * element_size)
        + num_post * (n * hidden * element_size + n * mhc_mult * 4 + n * mhc_mult * mhc_mult * 4 + n * mhc_mult * hidden * element_size)
    )
    return io_ref, io_fused


@pytest.mark.parametrize('num_layers,num_post,hidden', _CORRECTNESS_CASES)
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_mhc_multilayer_recompute_correctness(num_layers: int, num_post: int, hidden: int, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list, layer_input_list, residual_list = (
        generate_multilayer_recompute_test_data(1, 8192, 4, hidden, num_layers, num_post, dtype=dtype)
    )
    layer_input_ref, residual_ref = _mhc_multilayer_recompute_ref(initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list)
    mhc_multilayer_recompute(initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list, layer_input_list, residual_list)

    for i in range(num_layers):
        assert torch.equal(layer_input_list[i], layer_input_ref[i]), (
            f'layer_input[{i}] mismatch! max diff = {(layer_input_list[i].float() - layer_input_ref[i].float()).abs().max().item()}'
        )
    for i in range(num_post):
        assert torch.equal(residual_list[i], residual_ref[i]), (
            f'residual[{i}] mismatch! max diff = {(residual_list[i].float() - residual_ref[i].float()).abs().max().item()}'
        )


@pytest.mark.benchmark
@pytest.mark.parametrize('num_layers,num_post,bs,seq,mhc_mult,hidden', _BENCH_CASES)
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_mhc_multilayer_recompute_benchmark(
    num_layers: int,
    num_post: int,
    bs: int,
    seq: int,
    mhc_mult: int,
    hidden: int,
    dtype: torch.dtype,
    benchmark_record,
    benchmark_timer,
) -> None:
    torch.manual_seed(0)
    n = bs * seq
    initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list, layer_input_list, residual_list = (
        generate_multilayer_recompute_test_data(bs, seq, mhc_mult, hidden, num_layers, num_post, dtype=dtype)
    )
    io_ref, io_fused = _compute_io_bytes(n, mhc_mult, hidden, num_layers, num_post, initial_residual.element_size())
    theory = io_ref / io_fused

    def fn_ref() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return _mhc_multilayer_recompute_ref(initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list)

    def fn_fused() -> None:
        mhc_multilayer_recompute(initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list, layer_input_list, residual_list)

    fn_ref()
    fn_fused()
    t_ref_us = benchmark_timer(fn_ref)
    t_fused_us = benchmark_timer(fn_fused)
    speedup = t_ref_us / t_fused_us
    bw_ref_gbs = io_ref / t_ref_us / 1e3
    bw_fused_gbs = io_fused / t_fused_us / 1e3

    benchmark_record(
        kernel='mhc_multilayer_recompute',
        operation='recompute',
        params={'num_layers': num_layers, 'num_post': num_post, 'bs': bs, 'seq': seq, 'mhc_mult': mhc_mult, 'hidden': hidden, 'dtype': str(dtype)},
        time_us=t_fused_us,
        bandwidth_gbs=bw_fused_gbs,
        extras={
            'ref_time_us': t_ref_us,
            'speedup': speedup,
            'bw_ref_gbs': bw_ref_gbs,
            'theory': theory,
            'efficiency': speedup / theory,
        },
    )

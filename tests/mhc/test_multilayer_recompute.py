import pytest
import torch
from tile_kernels.modeling.mhc.ops.multilayer_recompute import mhc_multilayer_recompute
from tile_kernels.modeling.mhc.ops.post import mhc_post
from tile_kernels.modeling.mhc.ops.pre_apply_mix import mhc_pre_apply_mix
from tests.conftest import IS_HIP

# mhc_multilayer_recompute depends on mhc_post (PDL) and mhc_pre_apply_mix which crash on HIP/AMD
pytestmark = pytest.mark.skipif(IS_HIP, reason='mhc_multilayer_recompute depends on kernels with NV SM90+ features not supported on HIP/AMD')


_CORRECTNESS_CASES = [
    (1, 1, 2560),
    (3, 2, 2560),
    (3, 3, 2560),
    (10, 9, 2560),
    (10, 10, 2560),
    (10, 9, 4096),
    (10, 10, 4096),
    (10, 9, 7168),
    (10, 10, 7168),
    (10, 9, 8192),
    (10, 10, 8192),
]

_BENCH_CASES = [
    (10, 9, 1, 8192, 4, 2560),
    (10, 10, 1, 8192, 4, 2560),
    (10, 9, 1, 8192, 4, 4096),
    (10, 10, 1, 8192, 4, 4096),
    (10, 9, 1, 8192, 4, 7168),
    (10, 10, 1, 8192, 4, 7168),
    (10, 9, 1, 8192, 4, 8192),
    (10, 10, 1, 8192, 4, 8192),
]


def generate_multilayer_recompute_test_data(
    bs: int,
    seq: int,
    mhc_mult: int,
    hidden: int,
    num_layers: int,
    num_post: int,
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
]:
    initial_residual = torch.randn(bs, seq, mhc_mult, hidden, device='cuda', dtype=torch.bfloat16)
    pre_mix_list = [torch.randn(bs, seq, mhc_mult, 1, device='cuda', dtype=torch.float32) for _ in range(num_layers)]
    layer_output_list = [torch.randn(bs, seq, hidden, device='cuda', dtype=torch.bfloat16) for _ in range(num_post)]
    post_mix_list = [torch.randn(bs, seq, mhc_mult, 1, device='cuda', dtype=torch.float32) for _ in range(num_post)]
    comb_mix_list = [torch.randn(bs, seq, mhc_mult, mhc_mult, device='cuda', dtype=torch.float32) for _ in range(num_post)]
    layer_input_list = [torch.empty(bs, seq, hidden, device='cuda', dtype=torch.bfloat16) for _ in range(num_layers)]
    residual_list = [torch.empty(bs, seq, mhc_mult, hidden, device='cuda', dtype=torch.bfloat16) for _ in range(num_post)]
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
    for i in range(len(pre_mix_list)):
        layer_input = mhc_pre_apply_mix(residual, pre_mix_list[i])
        layer_input_refs.append(layer_input)
        if i < len(layer_output_list):
            residual = mhc_post(layer_output_list[i], residual, post_mix_list[i], comb_mix_list[i])
            residual_refs.append(residual)
    return layer_input_refs, residual_refs


def _compute_io_bytes(n: int, mhc_mult: int, hidden: int, num_layers: int, num_post: int) -> tuple[int, int]:
    io_pre = n * mhc_mult * hidden * 2 + n * mhc_mult * 4 + n * hidden * 2
    io_post = n * hidden * 2 + n * mhc_mult * hidden * 2 + n * mhc_mult * 4 + n * mhc_mult * mhc_mult * 4 + n * mhc_mult * hidden * 2
    io_ref = num_layers * io_pre + num_post * io_post
    io_fused = (
        n * mhc_mult * hidden * 2
        + num_layers * (n * mhc_mult * 4 + n * hidden * 2)
        + num_post * (n * hidden * 2 + n * mhc_mult * 4 + n * mhc_mult * mhc_mult * 4 + n * mhc_mult * hidden * 2)
    )
    return io_ref, io_fused


@pytest.mark.parametrize('num_layers,num_post,hidden', _CORRECTNESS_CASES)
def test_mhc_multilayer_recompute_correctness(num_layers: int, num_post: int, hidden: int) -> None:
    torch.manual_seed(0)
    initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list, layer_input_list, residual_list = (
        generate_multilayer_recompute_test_data(1, 8192, 4, hidden, num_layers, num_post)
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
def test_mhc_multilayer_recompute_benchmark(
    num_layers: int,
    num_post: int,
    bs: int,
    seq: int,
    mhc_mult: int,
    hidden: int,
    benchmark_record,
    benchmark_timer,
) -> None:
    torch.manual_seed(0)
    n = bs * seq
    io_ref, io_fused = _compute_io_bytes(n, mhc_mult, hidden, num_layers, num_post)
    theory = io_ref / io_fused
    initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list, layer_input_list, residual_list = (
        generate_multilayer_recompute_test_data(bs, seq, mhc_mult, hidden, num_layers, num_post)
    )

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
        params={'num_layers': num_layers, 'num_post': num_post, 'bs': bs, 'seq': seq, 'mhc_mult': mhc_mult, 'hidden': hidden},
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

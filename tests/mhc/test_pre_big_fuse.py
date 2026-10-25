from collections.abc import Callable

import pytest
import torch

from tile_kernels.config import get_device, is_ascend
from tile_kernels.mhc.norm_fn_kernel import mhc_reduce_partials_and_rmsnorm_fwd
from tile_kernels.mhc.pre_apply_mix_kernel import mhc_pre_apply_mix_fwd
from tile_kernels.mhc.pre_big_fuse_kernel import mhc_pre_big_fuse_fwd
from tile_kernels.mhc.pre_split_mixes_kernel import mhc_pre_split_mixes_fwd
from tile_kernels.mhc.sinkhorn_kernel import mhc_sinkhorn_fwd
from tile_kernels.rand import randn
from tile_kernels.testing.numeric import count_bytes


def generate_big_fuse_test_data(
    n1: int,
    mhc_mult: int,
    hidden_size: int,
    rms_eps: float = 1e-6,
    mhc_pre_eps: float = 1e-6,
    mhc_sinkhorn_eps: float = 1e-6,
    mhc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 10,
    n_splits: int = 16,
) -> dict[str, torch.Tensor | float]:
    n0 = 1
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2
    device = get_device()

    residual = (
        randn((n0, n1, mhc_mult, hidden_size), dtype=torch.float, device=device)
        .mul(1 + torch.arange(mhc_mult, device=device).mul(0.01).view(1, 1, -1, 1))
        .bfloat16()
    )

    fn = (
        randn((mhc_mult3, mhc_mult, hidden_size), dtype=torch.float, device=device)
        * 1e-4
        * (1 + torch.arange(mhc_mult, device=device).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)

    mhc_scale = randn((3,), dtype=torch.float, device=device) * 0.1
    mhc_base = randn((mhc_mult3,), dtype=torch.float, device=device) * 0.1

    return {
        'residual': residual,
        'fn': fn,
        'mhc_scale': mhc_scale,
        'mhc_base': mhc_base,
        'rms_eps': rms_eps,
        'mhc_pre_eps': mhc_pre_eps,
        'mhc_sinkhorn_eps': mhc_sinkhorn_eps,
        'mhc_post_mult_value': mhc_post_mult_value,
        'sinkhorn_repeat': sinkhorn_repeat,
        'n_splits': n_splits,
    }


def make_gemm_partials(residual: torch.Tensor, fn: torch.Tensor, n_splits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the (gemm_out_mul, gemm_out_sqrsum) inputs of the fused kernel from the torch reference.

    The prenorm GEMM itself lives in DeepGEMM now, so its outputs are
    reproduced here: split-K partials of residual @ fn.T plus the squared sum
    of the residual. Ascend reduces split-K inside the GEMM, so a single
    partial is produced there.
    """
    mhc_mult3, mhc_hidden_size = fn.shape
    num_tokens = residual.shape[0] * residual.shape[1]
    actual_n_splits = 1 if is_ascend() else n_splits
    split_size = mhc_hidden_size // actual_n_splits

    x_flat = residual.float().flatten(2, 3).reshape(num_tokens, mhc_hidden_size)
    out_mul = torch.empty(actual_n_splits, num_tokens, mhc_mult3, dtype=torch.float32, device=residual.device)
    sqrsum = torch.empty(actual_n_splits, num_tokens, dtype=torch.float32, device=residual.device)
    for i in range(actual_n_splits):
        x_split = x_flat[:, i * split_size : (i + 1) * split_size].unsqueeze(1)
        fn_split = fn[:, i * split_size : (i + 1) * split_size].unsqueeze(1)
        out_mul[i] = torch.einsum('mbk,nbk->mbn', x_split, fn_split).reshape(num_tokens, mhc_mult3)
        sqrsum[i] = x_split.square().sum(-1).reshape(num_tokens)
    return out_mul, sqrsum


def make_big_fuse_runner(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    test_data: dict[str, torch.Tensor | float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Callable[[], None]]:
    """Allocate the fused kernel outputs and return a callable running only the fused kernel."""
    residual = test_data['residual']
    assert isinstance(residual, torch.Tensor)
    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    num_tokens = residual.shape[0] * residual.shape[1]
    device = residual.device

    post_mix = torch.empty(*residual.shape[:-2], mhc_mult, 1, dtype=torch.float32, device=device)
    comb_mix = torch.empty(*residual.shape[:-2], mhc_mult, mhc_mult, dtype=torch.float32, device=device)
    layer_input = torch.empty(*residual.shape[:-2], hidden_size, dtype=torch.bfloat16, device=device)

    def _run() -> None:
        mhc_pre_big_fuse_fwd(
            gemm_out_mul,
            gemm_out_sqrsum,
            test_data['mhc_scale'],
            test_data['mhc_base'],
            residual.view(num_tokens, mhc_mult, hidden_size),
            post_mix.view(num_tokens, mhc_mult),
            comb_mix.view(num_tokens, mhc_mult * mhc_mult),
            layer_input.view(num_tokens, hidden_size),
            rms_eps=test_data['rms_eps'],
            mhc_pre_eps=test_data['mhc_pre_eps'],
            mhc_sinkhorn_eps=test_data['mhc_sinkhorn_eps'],
            mhc_post_mult_value=test_data['mhc_post_mult_value'],
            sinkhorn_repeat=test_data['sinkhorn_repeat'],
            n_splits=gemm_out_mul.shape[0],
        )

    return post_mix, comb_mix, layer_input, _run


def big_fuse_reference(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    test_data: dict[str, torch.Tensor | float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unfused op chain (reduce/rmsnorm + split mixes + sinkhorn + apply mix) on the same GEMM outputs."""
    residual = test_data['residual']
    assert isinstance(residual, torch.Tensor)
    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2
    num_tokens = residual.shape[0] * residual.shape[1]
    n_splits = gemm_out_mul.shape[0]
    device = residual.device

    out_mul = torch.empty(num_tokens, 1, mhc_mult3, dtype=torch.float32, device=device)
    sqrsum = torch.empty(num_tokens, 1, dtype=torch.float32, device=device)
    mixes = torch.empty(num_tokens, mhc_mult3, dtype=torch.float32, device=device)
    mhc_reduce_partials_and_rmsnorm_fwd(
        gemm_out_mul.view(n_splits, num_tokens, 1, mhc_mult3),
        gemm_out_sqrsum.view(n_splits, num_tokens, 1),
        out_mul,
        sqrsum,
        mixes,
        rms_group_size=mhc_mult * hidden_size,
        rms_eps=test_data['rms_eps'],
        n_splits=n_splits,
    )

    pre_mix = torch.empty(num_tokens, mhc_mult, dtype=torch.float32, device=device)
    post_mix = torch.empty(num_tokens, mhc_mult, dtype=torch.float32, device=device)
    comb_mix = torch.empty(num_tokens, mhc_mult2, dtype=torch.float32, device=device)
    mhc_pre_split_mixes_fwd(
        mixes,
        test_data['mhc_scale'],
        test_data['mhc_base'],
        pre_mix,
        post_mix,
        comb_mix,
        test_data['mhc_post_mult_value'],
        test_data['mhc_pre_eps'],
    )

    comb_res_mix = comb_mix.view(num_tokens, mhc_mult, mhc_mult)
    comb_res_mix_norm = torch.empty_like(comb_res_mix)
    mhc_sinkhorn_fwd(comb_res_mix, comb_res_mix_norm, test_data['sinkhorn_repeat'], test_data['mhc_sinkhorn_eps'])

    layer_input = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    mhc_pre_apply_mix_fwd(residual.view(num_tokens, mhc_mult, hidden_size), pre_mix, layer_input)

    outer_shape = residual.shape[:-2]
    return (
        post_mix.view(*outer_shape, mhc_mult, 1),
        comb_res_mix_norm.view(*outer_shape, mhc_mult, mhc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


@pytest.mark.parametrize('n1', [512, 1024, 2048, 8192, 509, 1021])
@pytest.mark.parametrize('hidden_size', [1280, 2560, 4096, 4096 + 64, 5120])
@pytest.mark.parametrize('mhc_mult', [4])
def test_correctness(
    n1: int,
    hidden_size: int,
    mhc_mult: int,
) -> None:
    test_data = generate_big_fuse_test_data(
        n1=n1,
        mhc_mult=mhc_mult,
        hidden_size=hidden_size,
    )

    gemm_out_mul, gemm_out_sqrsum = make_gemm_partials(test_data['residual'], test_data['fn'], test_data['n_splits'])

    post_mix_fused, comb_mix_fused, layer_input_fused, _run = make_big_fuse_runner(gemm_out_mul, gemm_out_sqrsum, test_data)
    _run()

    post_mix_ref, comb_mix_ref, layer_input_ref = big_fuse_reference(gemm_out_mul, gemm_out_sqrsum, test_data)

    assert torch.equal(post_mix_fused, post_mix_ref)
    assert torch.equal(comb_mix_fused, comb_mix_ref)
    if torch.version.hip is None:
        assert torch.equal(layer_input_fused, layer_input_ref)
    else:
        # HIP computes the apply-mix in registers (see pre_big_fuse_cuda.py) with another thread
        # layout than mhc_pre_apply_mix_fwd, so a few outputs round one bf16 ulp apart.
        torch.testing.assert_close(layer_input_fused, layer_input_ref, rtol=2**-7, atol=2**-7)


_PRE_BIG_FUSE_BENCH_CASES = [
    (512, 4096),
    (512, 7168),
    (8192, 4096),
    (8192, 5120),
    (8192, 7168),
]


@pytest.mark.benchmark
@pytest.mark.parametrize('n1,hidden_size', _PRE_BIG_FUSE_BENCH_CASES)
def test_mhc_pre_big_fuse_fwd_benchmark(
    n1: int,
    hidden_size: int,
    benchmark_timer,
    benchmark_record,
) -> None:
    mhc_mult = 4
    test_data = generate_big_fuse_test_data(n1=n1, mhc_mult=mhc_mult, hidden_size=hidden_size)
    residual = test_data['residual']
    fn = test_data['fn']
    mhc_scale = test_data['mhc_scale']
    mhc_base = test_data['mhc_base']

    # Precompute the GEMM outputs with the torch reference and benchmark only
    # the fused pre_big_fuse kernel: the GEMM op itself now lives in DeepGEMM.
    gemm_out_mul, gemm_out_sqrsum = make_gemm_partials(residual, fn, test_data['n_splits'])
    post_mix, comb_mix, layer_input, _run = make_big_fuse_runner(gemm_out_mul, gemm_out_sqrsum, test_data)

    io_gb = (
        count_bytes(
            gemm_out_mul,
            gemm_out_sqrsum,
            mhc_scale,
            mhc_base,
            residual.view(-1, mhc_mult, hidden_size),
            post_mix,
            comb_mix,
            layer_input,
        )
        / 1e9
    )

    _run()
    t_us = benchmark_timer(_run)

    benchmark_record(
        kernel='mhc_pre_big_fuse_fwd',
        operation='pre_big_fuse_forward',
        params={'n1': n1, 'hidden_size': hidden_size, 'mhc_mult': mhc_mult},
        time_us=t_us,
        bandwidth_gbs=io_gb / t_us * 1e6,
        extras={'io_gb': io_gb, 'sinkhorn_repeat': test_data['sinkhorn_repeat']},
    )

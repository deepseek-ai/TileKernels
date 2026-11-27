from contextlib import contextmanager

import pytest
import torch

from tile_kernels.config import get_device, is_ascend
from tile_kernels.mhc.norm_fn_kernel import (
    default_n_splits,
    mhc_fn_normw_merge_bwd,
    mhc_fn_normw_merge_fwd,
    mhc_reduce_partials_and_rmsnorm_bwd,
    mhc_reduce_partials_and_rmsnorm_fwd,
)
from tile_kernels.rand import randn, randn_like
from tile_kernels.testing.generator import generate_hidden_sizes, get_test_level
from tile_kernels.testing.numeric import count_bytes
from tile_kernels.torch.mhc import mhc_pre_norm_fn_partials_ref, mhc_pre_norm_fn_ref


@contextmanager
def _reference_matmul_precision():
    if is_ascend():
        old_allow_hf32 = torch.npu.matmul.allow_hf32
        torch.npu.matmul.allow_hf32 = True
        try:
            yield
        finally:
            torch.npu.matmul.allow_hf32 = old_allow_hf32
    else:
        old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32


def generate_norm_fn_test_data(
    n1: int,
    mhc_mult: int,
    hidden_size: int,
    generate_normw: bool,
) -> dict[str, torch.Tensor]:
    n0 = 1
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    mhc_hidden_size = mhc_mult * hidden_size
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

    if generate_normw:
        normw = randn((mhc_hidden_size,), dtype=torch.float, device=device) * 0.1 + 1.0
    else:
        normw = None

    out_grad = randn((n0, n1, mhc_mult3), dtype=torch.float, device=device)

    return {
        'residual': residual,
        'fn': fn,
        'normw': normw,
        'out_grad': out_grad,
        'mhc_norm_eps': 1e-6,
    }


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    # Keep the reduction on the accelerator. This form is algebraically
    # equivalent to 1 - 2 * <x, y> / (||x||^2 + ||y||^2), but avoids the
    # cancellation in '1 - sim' and is therefore accurate enough in fp32.
    x, y = x.float(), y.float()
    delta = x - y
    numerator = (delta * delta).sum()
    denominator = (x * x + y * y).sum()
    diff = torch.where(denominator == 0, 0.0, numerator / denominator)
    return float(diff)


def _assert_calc_diff_close(x: torch.Tensor, y: torch.Tensor, tol: float = 1e-8) -> None:
    diff = _calc_diff(x, y)
    assert diff < tol, f'calc_diff={diff:.10g} exceeds tolerance {tol}'


def _ref_reduce_rmsnorm(
    out_mul: torch.Tensor,
    sqrsum: torch.Tensor,
    mhc_hidden_size: int,
    eps: float,
) -> torch.Tensor:
    """Torch tail of mhc_pre_norm_fn_ref: rms-normalize and reduce over the RMS group."""
    rms = (sqrsum / mhc_hidden_size + eps).rsqrt()
    return (out_mul * rms.unsqueeze(-1)).sum(-2)


@pytest.mark.parametrize('n1', [4096, 8192, 509, 8191])
@pytest.mark.parametrize('hidden_size', generate_hidden_sizes(max(get_test_level(), 1), ['main']))
@pytest.mark.parametrize('generate_normw', [False, True])
def test_correctness(
    n1: int,
    hidden_size: int,
    generate_normw: bool,
) -> None:
    """Cover the fn x normw merge and the reduce/rmsnorm operators (fwd + bwd).

    The prenorm GEMM moved to DeepGEMM, so its outputs are built from the
    torch reference and only the two remaining norm-fn operators are
    exercised here. n1=509 and 8191 are not multiples of the kernel token block
    (16, or 32 for the large-M forward), so they cover the ragged trailing block.
    """
    mhc_mult = 4
    mhc_hidden_size = mhc_mult * hidden_size
    mhc_mult3 = mhc_mult * (2 + mhc_mult)

    test_data = generate_norm_fn_test_data(
        n1=n1,
        mhc_mult=mhc_mult,
        hidden_size=hidden_size,
        generate_normw=generate_normw,
    )
    residual = test_data['residual']
    fn = test_data['fn']
    normw = test_data['normw']
    out_grad = test_data['out_grad']
    mhc_norm_eps = test_data['mhc_norm_eps']
    num_tokens = residual.shape[0] * residual.shape[1]

    actual_n_splits = default_n_splits(mhc_hidden_size)
    merged_fn = fn
    if normw is not None:
        # fn x normw merge, compared against the merge that mhc_pre_norm_fn_ref
        # performs in torch before its GEMM.
        merged_fn = torch.empty_like(fn)
        mhc_fn_normw_merge_fwd(fn, normw, merged_fn)
        _assert_calc_diff_close(merged_fn, fn * normw)

        merged_fn_grad = randn_like(fn)
        fn_grad = torch.zeros_like(fn)
        normw_grad = torch.zeros_like(normw)
        mhc_fn_normw_merge_bwd(fn, normw, merged_fn_grad, fn_grad, normw_grad)
        _assert_calc_diff_close(fn_grad, merged_fn_grad * normw)
        _assert_calc_diff_close(normw_grad, (merged_fn_grad * fn).sum(0))

    with _reference_matmul_precision():
        mixes_ref = mhc_pre_norm_fn_ref(residual, fn, normw, mhc_norm_eps)
        gemm_out_mul, gemm_out_sqrsum = mhc_pre_norm_fn_partials_ref(residual, merged_fn, actual_n_splits)

    out_mul = torch.empty(num_tokens, 1, mhc_mult3, dtype=torch.float32, device=residual.device)
    sqrsum = torch.empty(num_tokens, 1, dtype=torch.float32, device=residual.device)
    mixes_tl = torch.empty(num_tokens, mhc_mult3, dtype=torch.float32, device=residual.device)
    mhc_reduce_partials_and_rmsnorm_fwd(
        gemm_out_mul,
        gemm_out_sqrsum,
        out_mul,
        sqrsum,
        mixes_tl,
        rms_group_size=mhc_hidden_size,
        rms_eps=mhc_norm_eps,
        n_splits=actual_n_splits,
    )
    torch.testing.assert_close(mixes_tl.view_as(mixes_ref), mixes_ref, atol=1e-3, rtol=1e-3)

    # The reduce/rmsnorm backward only sees the fwd outputs: feed it the kernel
    # outputs and compare against autograd through the torch reference tail.
    out_mul_grad = torch.empty_like(out_mul)
    sqrsum_grad = torch.empty_like(sqrsum)
    mhc_reduce_partials_and_rmsnorm_bwd(
        out_grad.view(-1, mhc_mult3),
        out_mul,
        sqrsum,
        out_mul_grad,
        sqrsum_grad,
        rms_group_size=mhc_hidden_size,
        rms_eps=mhc_norm_eps,
    )

    out_mul_ref = out_mul.clone().requires_grad_()
    sqrsum_ref = sqrsum.clone().requires_grad_()
    mixes_ref_tail = _ref_reduce_rmsnorm(out_mul_ref, sqrsum_ref, mhc_hidden_size, mhc_norm_eps)
    torch.autograd.backward([mixes_ref_tail], [out_grad.view(-1, mhc_mult3)])
    assert out_mul_ref.grad is not None
    assert sqrsum_ref.grad is not None
    _assert_calc_diff_close(out_mul_grad, out_mul_ref.grad, tol=1e-8)
    _assert_calc_diff_close(sqrsum_grad, sqrsum_ref.grad, tol=1e-8)


_NORMW_MERGE_BENCH_CASES = generate_hidden_sizes(1, ['4*main'])


@pytest.mark.benchmark
@pytest.mark.parametrize('hidden', _NORMW_MERGE_BENCH_CASES)
def test_fn_normw_merge_fwd_benchmark(
    hidden: int,
    benchmark_timer,
    benchmark_record,
) -> None:
    device = get_device()
    mhc_mult3 = 24
    fn = randn(mhc_mult3, hidden, dtype=torch.float32, device=device)
    normw = randn(hidden, dtype=torch.float32, device=device)
    out_fn = torch.empty_like(fn)

    def _run():
        mhc_fn_normw_merge_fwd(fn, normw, out_fn)

    _run()
    t_us = benchmark_timer(_run)
    io_gb = count_bytes(fn, normw, out_fn) / 1e9
    benchmark_record(
        kernel='mhc_fn_normw_merge_fwd',
        operation='normw_merge_forward',
        params={'hidden': hidden},
        time_us=t_us,
        bandwidth_gbs=io_gb / t_us * 1e6,
        extras={'io_gb': io_gb, 'mhc_mult3': mhc_mult3},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('hidden', _NORMW_MERGE_BENCH_CASES)
def test_fn_normw_merge_bwd_benchmark(
    hidden: int,
    benchmark_timer,
    benchmark_record,
) -> None:
    device = get_device()
    mhc_mult3 = 24
    fn = randn(mhc_mult3, hidden, dtype=torch.float32, device=device)
    normw = randn(hidden, dtype=torch.float32, device=device)
    out_fn_grad = randn_like(fn)
    fn_grad = randn_like(fn)
    normw_grad = randn_like(normw)

    def _run():
        mhc_fn_normw_merge_bwd(
            fn,
            normw,
            out_fn_grad,
            fn_grad,
            normw_grad,
        )

    _run()
    t_us = benchmark_timer(_run)
    io_gb = (
        count_bytes(
            fn,
            normw,
            out_fn_grad,
            fn_grad,
            fn_grad,
            normw_grad,
            normw_grad,
        )
        / 1e9
    )
    benchmark_record(
        kernel='mhc_fn_normw_merge_bwd',
        operation='normw_merge_backward',
        params={'hidden': hidden},
        time_us=t_us,
        bandwidth_gbs=io_gb / t_us * 1e6,
        extras={'io_gb': io_gb, 'mhc_mult3': mhc_mult3, 'grad_buffers_inout': True},
    )

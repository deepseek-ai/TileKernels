import pytest
import torch

from tile_kernels.config import get_device
from tile_kernels.mhc.sinkhorn_kernel import mhc_sinkhorn_bwd, mhc_sinkhorn_fwd
from tile_kernels.rand import randn
from tile_kernels.testing.numeric import count_bytes
from tile_kernels.torch.mhc import sinkhorn_normalize_ref


def generate_sinkhorn_test_data(n0: int, n1: int, mhc: int, device: str | None = None) -> dict[str, torch.Tensor]:
    device = get_device() if device is None else device
    comb_res_mix = randn((n0, n1, mhc, mhc), dtype=torch.float32, device=device)
    out_grad = randn((n0, n1, mhc, mhc), dtype=torch.float32, device=device)

    return {
        'comb_res_mix': comb_res_mix,
        'out_grad': out_grad,
        'repeat': 20,
        'eps': 1e-6,
    }


def _run_sinkhorn_fwd(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    x_flat = x.contiguous().view(-1, *x.shape[-2:])
    output = torch.empty_like(x_flat)
    mhc_sinkhorn_fwd(x_flat, output, repeat, eps)
    return output.view_as(x)


def _run_sinkhorn_bwd(out_grad: torch.Tensor, x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    x_flat = x.contiguous().view(-1, *x.shape[-2:])
    out_grad_flat = out_grad.contiguous().view(-1, *out_grad.shape[-2:])
    grad_input = torch.empty_like(x_flat)
    mhc_sinkhorn_bwd(out_grad_flat, x_flat, grad_input, repeat, eps)
    return grad_input.view_as(out_grad)


@pytest.mark.parametrize('n0', [1, 2])
@pytest.mark.parametrize('n1', [1, 1024, 4096])
@pytest.mark.parametrize('mhc', [4])
def test_sinkhorn_comprehensive(n0: int, n1: int, mhc: int) -> None:
    test_data = generate_sinkhorn_test_data(n0=n0, n1=n1, mhc=mhc)

    out_tl = _run_sinkhorn_fwd(test_data['comb_res_mix'], test_data['repeat'], test_data['eps'])
    grad_tl = _run_sinkhorn_bwd(
        test_data['out_grad'],
        test_data['comb_res_mix'],
        test_data['repeat'],
        test_data['eps'],
    )

    comb_res_mix_ref = test_data['comb_res_mix'].clone().requires_grad_()
    out_ref = sinkhorn_normalize_ref(comb_res_mix_ref, test_data['repeat'], test_data['eps'])
    torch.autograd.backward([out_ref], [test_data['out_grad']])
    assert comb_res_mix_ref.grad is not None

    torch.testing.assert_close(out_tl, out_ref)
    torch.testing.assert_close(grad_tl, comb_res_mix_ref.grad)


@pytest.mark.parametrize('offset', [100.0, -100.0])
@pytest.mark.parametrize('eps', [1e-6, 1e-2])
def test_sinkhorn_large_same_sign_logits(offset: float, eps: float) -> None:
    device = get_device()
    comb_res_mix = (
        torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [3.0, 1.0, 2.0, 0.0],
                [1.5, 0.5, 3.5, 2.5],
                [2.5, 3.5, 0.5, 1.5],
            ],
            dtype=torch.float32,
            device=device,
        ).view(1, 1, 4, 4)
        + offset
    )
    out_grad = torch.tensor(
        [
            [0.5, -1.0, 2.0, -0.5],
            [1.5, 0.25, -0.75, 0.5],
            [-1.5, 2.5, 0.75, -2.0],
            [1.0, -0.25, 1.25, -1.75],
        ],
        dtype=torch.float32,
        device=device,
    ).view(1, 1, 4, 4)
    repeat = 20

    out_tl = _run_sinkhorn_fwd(comb_res_mix, repeat, eps)
    grad_tl = _run_sinkhorn_bwd(out_grad, comb_res_mix, repeat, eps)

    comb_res_mix_ref = comb_res_mix.clone().requires_grad_()
    out_ref = sinkhorn_normalize_ref(comb_res_mix_ref, repeat, eps)
    torch.autograd.backward([out_ref], [out_grad])
    assert comb_res_mix_ref.grad is not None

    assert torch.isfinite(out_tl).all()
    assert torch.isfinite(grad_tl).all()
    torch.testing.assert_close(out_tl, out_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(grad_tl, comb_res_mix_ref.grad, rtol=1e-5, atol=1e-6)


_SINKHORN_BENCH_CASES = [
    (1, 512, 4),
    (1, 8192, 4),
]


@pytest.mark.benchmark
@pytest.mark.parametrize('n0,n1,mhc', _SINKHORN_BENCH_CASES)
def test_sinkhorn_fwd_benchmark(
    n0: int,
    n1: int,
    mhc: int,
    benchmark_timer,
    benchmark_record,
) -> None:
    test_data = generate_sinkhorn_test_data(n0=n0, n1=n1, mhc=mhc)
    x = test_data['comb_res_mix']
    repeat = test_data['repeat']
    eps = test_data['eps']
    num_tokens = n0 * n1

    x_flat = x.contiguous().view(-1, *x.shape[-2:])
    output = torch.empty_like(x_flat)
    io_gb = count_bytes(x) * 2 / 1e9  # read + write, same shape

    t_us = benchmark_timer(lambda: mhc_sinkhorn_fwd(x_flat, output, repeat, eps))

    benchmark_record(
        kernel='mhc_sinkhorn',
        operation='sinkhorn_forward',
        params={'n0': n0, 'n1': n1, 'mhc': mhc, 'repeat': repeat},
        time_us=t_us,
        extras={'io_gb': io_gb, 'num_tokens': num_tokens},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('n0,n1,mhc', _SINKHORN_BENCH_CASES)
def test_sinkhorn_bwd_benchmark(
    n0: int,
    n1: int,
    mhc: int,
    benchmark_timer,
    benchmark_record,
) -> None:
    test_data = generate_sinkhorn_test_data(n0=n0, n1=n1, mhc=mhc)
    x = test_data['comb_res_mix']
    repeat = test_data['repeat']
    eps = test_data['eps']
    out_grad = test_data['out_grad']
    num_tokens = n0 * n1

    x_flat = x.contiguous().view(-1, *x.shape[-2:])
    out_grad_flat = out_grad.contiguous().view(-1, *out_grad.shape[-2:])
    grad_input = torch.empty_like(x_flat)

    io_gb = count_bytes(x) * 2 / 1e9  # read + write, same shape

    t_us = benchmark_timer(lambda: mhc_sinkhorn_bwd(out_grad_flat, x_flat, grad_input, repeat, eps))

    benchmark_record(
        kernel='mhc_sinkhorn',
        operation='sinkhorn_backward',
        params={'n0': n0, 'n1': n1, 'mhc': mhc, 'repeat': repeat},
        time_us=t_us,
        extras={'io_gb': io_gb, 'num_tokens': num_tokens},
    )

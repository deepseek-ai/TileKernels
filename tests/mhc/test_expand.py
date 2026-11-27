import pytest
import torch

from tile_kernels.config import get_device, is_ascend
from tile_kernels.mhc.expand_kernel import expand_to_mhc_bwd, expand_to_mhc_fwd
from tile_kernels.rand import randn
from tile_kernels.testing.generator import generate_hidden_sizes, get_test_level
from tile_kernels.testing.numeric import count_bytes
from tile_kernels.torch.mhc import expand_to_mhc_ref


def generate_expand_test_data(
    n0: int, n1: int, mhc_mult: int, h: int, device: str | None = None, dtype: torch.dtype = torch.bfloat16
) -> dict[str, torch.Tensor]:
    device = get_device() if device is None else device
    x = randn(n0, n1, h, dtype=dtype, device=device)
    o_grad = randn(n0, n1, mhc_mult, h, dtype=dtype, device=device)

    return {'x': x, 'o_grad': o_grad, 'mhc_mult': mhc_mult}


def _run_expand_fwd(x: torch.Tensor, mhc_mult: int) -> torch.Tensor:
    out = x.new_empty(*x.shape[:-1], mhc_mult, x.shape[-1])
    assert x.is_contiguous()
    expand_to_mhc_fwd(x.flatten(0, -2), out.flatten(0, -3))
    return out


def _run_expand_bwd(o_grad: torch.Tensor) -> torch.Tensor:
    x_grad = o_grad.new_empty(*o_grad.shape[:-2], o_grad.shape[-1])
    expand_to_mhc_bwd(o_grad.flatten(0, -3), x_grad.flatten(0, -2))
    return x_grad


@pytest.mark.parametrize('n0', [1, 2])
@pytest.mark.parametrize('n1', [1024, 4096])
@pytest.mark.parametrize('mhc_mult', [4])
@pytest.mark.parametrize('h', generate_hidden_sizes(max(get_test_level(), 1), ['main'], align=128 if is_ascend() else 64))
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_expand_comprehensive(n0: int, n1: int, mhc_mult: int, h: int, dtype: torch.dtype) -> None:
    test_data = generate_expand_test_data(n0=n0, n1=n1, mhc_mult=mhc_mult, h=h, dtype=dtype)

    out_tl = _run_expand_fwd(test_data['x'], test_data['mhc_mult'])
    out_ref = expand_to_mhc_ref(test_data['x'], test_data['mhc_mult'])
    torch.testing.assert_close(out_tl, out_ref)

    x_grad_tl = _run_expand_bwd(test_data['o_grad'])

    x_ref = test_data['x'].clone().requires_grad_()
    out_ref = expand_to_mhc_ref(x_ref, test_data['mhc_mult'])
    torch.autograd.backward([out_ref], [test_data['o_grad']])
    assert x_ref.grad is not None

    torch.testing.assert_close(x_grad_tl, x_ref.grad)


_EXPAND_BENCH_CASES = [(1, n1, 4, h) for n1 in (512, 8192) for h in generate_hidden_sizes(1, ['main'])]


@pytest.mark.benchmark
@pytest.mark.parametrize('n0,n1,mhc_mult,h', _EXPAND_BENCH_CASES)
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_expand_fwd_benchmark(
    n0: int,
    n1: int,
    mhc_mult: int,
    h: int,
    dtype: torch.dtype,
    benchmark_timer,
    benchmark_record,
) -> None:
    num_tokens = n0 * n1
    device = get_device()
    x = randn(num_tokens, h, dtype=dtype, device=device)
    o = torch.empty(num_tokens, mhc_mult, h, dtype=dtype, device=device)

    io_gb = count_bytes(x, o) / 1e9

    t_us = benchmark_timer(lambda: expand_to_mhc_fwd(x, o))

    benchmark_record(
        kernel='expand_to_mhc_fwd',
        operation='expand_forward',
        params={'n0': n0, 'n1': n1, 'mhc_mult': mhc_mult, 'h': h, 'dtype': str(dtype)},
        time_us=t_us,
        bandwidth_gbs=io_gb / t_us * 1e6,
        extras={'io_gb': io_gb},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('n0,n1,mhc_mult,h', _EXPAND_BENCH_CASES)
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_expand_bwd_benchmark(
    n0: int,
    n1: int,
    mhc_mult: int,
    h: int,
    dtype: torch.dtype,
    benchmark_timer,
    benchmark_record,
) -> None:
    num_tokens = n0 * n1
    device = get_device()
    o_grad = randn(num_tokens, mhc_mult, h, dtype=dtype, device=device)
    x_grad = torch.empty(num_tokens, h, dtype=dtype, device=device)

    io_gb = count_bytes(o_grad, x_grad) / 1e9

    t_us = benchmark_timer(lambda: expand_to_mhc_bwd(o_grad, x_grad))

    benchmark_record(
        kernel='expand_to_mhc_bwd',
        operation='expand_backward',
        params={'n0': n0, 'n1': n1, 'mhc_mult': mhc_mult, 'h': h, 'dtype': str(dtype)},
        time_us=t_us,
        bandwidth_gbs=io_gb / t_us * 1e6,
        extras={'io_gb': io_gb},
    )

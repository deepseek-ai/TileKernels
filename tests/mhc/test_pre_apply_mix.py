import pytest
import torch

from tile_kernels.config import get_device
from tile_kernels.mhc.pre_apply_mix_kernel import mhc_pre_apply_mix_bwd, mhc_pre_apply_mix_fwd
from tile_kernels.rand import randn
from tile_kernels.testing.generator import generate_hidden_sizes, get_test_level
from tile_kernels.testing.numeric import count_bytes
from tile_kernels.torch.mhc import mhc_pre_apply_mix_ref


def generate_pre_apply_mix_test_data(
    n0: int, n1: int, mhc: int, h: int, device: str | None = None, dtype: torch.dtype = torch.bfloat16
) -> dict[str, torch.Tensor]:
    if device is None:
        device = get_device()
    x = randn(n0, n1, mhc, h, dtype=dtype, device=device).sigmoid()
    mix = randn(n0, n1, mhc, 1, dtype=torch.float32, device=device).softmax(-2)
    o_grad = randn(n0, n1, h, dtype=dtype, device=device)

    return {
        'x': x,
        'mix': mix,
        'o_grad': o_grad,
    }


@pytest.mark.parametrize('n0', [1, 2])
@pytest.mark.parametrize('n1', [1024, 4096])
@pytest.mark.parametrize('h', generate_hidden_sizes(max(get_test_level(), 1), ['main']))
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_pre_apply_mix_comprehensive(n0: int, n1: int, h: int, dtype: torch.dtype) -> None:
    mhc = 4

    test_data = generate_pre_apply_mix_test_data(n0=n0, n1=n1, mhc=mhc, h=h, dtype=dtype)
    x = test_data['x']
    mix = test_data['mix']
    o_grad = test_data['o_grad']
    num_tokens = n0 * n1

    o_tl = torch.empty(n0, n1, h, dtype=dtype, device=x.device)
    mhc_pre_apply_mix_fwd(
        x.view(num_tokens, mhc, h),
        mix.view(num_tokens, mhc),
        o_tl.view(num_tokens, h),
    )

    x_grad_tl = torch.zeros_like(x)
    mix_grad_tl = mhc_pre_apply_mix_bwd(
        o_grad.view(num_tokens, h),
        x.view(num_tokens, mhc, h),
        mix.view(num_tokens, mhc),
        x_grad_tl.view(num_tokens, mhc, h),
    ).view(n0, n1, mhc, 1)

    x_ref = x.clone().requires_grad_()
    mix_ref = mix.clone().requires_grad_()
    o_ref = mhc_pre_apply_mix_ref(x_ref, mix_ref)
    torch.autograd.backward([o_ref], [o_grad])
    assert x_ref.grad is not None
    assert mix_ref.grad is not None

    torch.testing.assert_close(o_tl, o_ref, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(x_grad_tl, x_ref.grad, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(mix_grad_tl, mix_ref.grad, atol=1e-2, rtol=1e-3)


_PRE_FWD_BENCH_CASES = [(1, n1, h, 4) for n1 in (512, 8192) for h in generate_hidden_sizes(1, ['main'])]


@pytest.mark.benchmark
@pytest.mark.parametrize('n0,n1,h,mhc_mult', _PRE_FWD_BENCH_CASES)
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_mhc_pre_apply_mix_fwd_benchmark(
    n0: int,
    n1: int,
    h: int,
    mhc_mult: int,
    dtype: torch.dtype,
    benchmark_timer,
    benchmark_record,
) -> None:
    test_data = generate_pre_apply_mix_test_data(n0=n0, n1=n1, mhc=mhc_mult, h=h, dtype=dtype)
    num_tokens = n0 * n1
    x = test_data['x'].reshape(num_tokens, mhc_mult, h)
    mix = test_data['mix'].reshape(num_tokens, mhc_mult)
    o = torch.empty((num_tokens, h), dtype=dtype, device=x.device)

    io_gb = count_bytes(x, mix, o) / 1e9

    t_us = benchmark_timer(lambda: mhc_pre_apply_mix_fwd(x, mix, o))

    benchmark_record(
        kernel='mhc_pre_apply_mix_fwd',
        operation='pre_apply_mix_forward',
        params={'n0': n0, 'n1': n1, 'h': h, 'mhc_mult': mhc_mult, 'dtype': str(dtype)},
        time_us=t_us,
        bandwidth_gbs=io_gb / t_us * 1e6,
        extras={'io_gb': io_gb},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('n0,n1,h,mhc_mult', _PRE_FWD_BENCH_CASES)
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_mhc_pre_apply_mix_bwd_benchmark(
    n0: int,
    n1: int,
    h: int,
    mhc_mult: int,
    dtype: torch.dtype,
    benchmark_timer,
    benchmark_record,
) -> None:
    test_data = generate_pre_apply_mix_test_data(n0=n0, n1=n1, mhc=mhc_mult, h=h, dtype=dtype)
    num_tokens = n0 * n1
    o_grad = test_data['o_grad'].reshape(num_tokens, h)
    x = test_data['x'].reshape(num_tokens, mhc_mult, h)
    mix = test_data['mix'].reshape(num_tokens, mhc_mult)
    x_grad = torch.zeros_like(x)

    mix_grad = mhc_pre_apply_mix_bwd(o_grad, x, mix, x_grad)
    io_gb = count_bytes(o_grad, x, mix, x_grad, x_grad, mix_grad) / 1e9

    t_us = benchmark_timer(lambda: mhc_pre_apply_mix_bwd(o_grad, x, mix, x_grad))

    benchmark_record(
        kernel='mhc_pre_apply_mix_bwd',
        operation='pre_apply_mix_backward',
        params={'n0': n0, 'n1': n1, 'h': h, 'mhc_mult': mhc_mult, 'dtype': str(dtype)},
        time_us=t_us,
        bandwidth_gbs=io_gb / t_us * 1e6,
        extras={'io_gb': io_gb},
    )

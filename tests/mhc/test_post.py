import pytest
import torch

from tile_kernels.config import get_device
from tile_kernels.mhc.post_kernel import mhc_post_bwd, mhc_post_fwd
from tile_kernels.rand import randn
from tile_kernels.testing.numeric import count_bytes
from tile_kernels.torch.mhc import mhc_post_ref


def generate_mhc_post_test_data(
    n0: int,
    n1: int,
    h: int,
    mhc_mult: int,
    device: str | None = None,
) -> dict[str, torch.Tensor]:
    if device is None:
        device = get_device()
    x = randn((n0, n1, h), dtype=torch.bfloat16, device=device)
    residual = randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16, device=device)
    post_layer_mix = randn((n0, n1, mhc_mult, 1), dtype=torch.float32, device=device)
    comb_res_mix = randn((n0, n1, mhc_mult, mhc_mult), dtype=torch.float32, device=device)

    o_grad = randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16, device=device)

    return {
        'x': x,
        'residual': residual,
        'post_layer_mix': post_layer_mix,
        'comb_res_mix': comb_res_mix,
        'o_grad': o_grad,
    }


@pytest.mark.parametrize('n0', [1, 2])
@pytest.mark.parametrize('n1', [4096])
@pytest.mark.parametrize('h', [1280, 2560, 5120, 7168])
def test_mhc_post_comprehensive(n0: int, n1: int, h: int) -> None:
    test_data = generate_mhc_post_test_data(n0=n0, n1=n1, h=h, mhc_mult=4)

    out_tl = mhc_post_fwd(
        test_data['x'],
        test_data['residual'],
        test_data['post_layer_mix'],
        test_data['comb_res_mix'],
    )
    grad_x_tl, grad_residual_tl, grad_post_layer_mix_tl, grad_comb_res_mix_tl = mhc_post_bwd(
        test_data['x'],
        test_data['residual'],
        test_data['post_layer_mix'],
        test_data['comb_res_mix'],
        test_data['o_grad'],
    )

    x_ref = test_data['x'].clone().requires_grad_()
    residual_ref = test_data['residual'].clone().requires_grad_()
    post_layer_mix_ref = test_data['post_layer_mix'].clone().requires_grad_()
    comb_res_mix_ref = test_data['comb_res_mix'].clone().requires_grad_()
    out_ref = mhc_post_ref(x_ref, residual_ref, post_layer_mix_ref, comb_res_mix_ref)
    torch.autograd.backward([out_ref], [test_data['o_grad']])
    assert x_ref.grad is not None
    assert residual_ref.grad is not None
    assert post_layer_mix_ref.grad is not None
    assert comb_res_mix_ref.grad is not None

    torch.testing.assert_close(out_tl, out_ref)
    torch.testing.assert_close(grad_x_tl, x_ref.grad)
    torch.testing.assert_close(grad_residual_tl, residual_ref.grad)
    torch.testing.assert_close(
        grad_post_layer_mix_tl,
        post_layer_mix_ref.grad,
        atol=1e-4,
        rtol=1e-4,
    )
    # The hidden-dim reduction behind this gradient is ordered differently on HIP.
    comb_tol = 2e-4 if torch.version.hip is not None else 1e-4
    torch.testing.assert_close(
        grad_comb_res_mix_tl,
        comb_res_mix_ref.grad,
        atol=comb_tol,
        rtol=comb_tol,
    )


_POST_FWD_BENCH_CASES = [
    (1, 512, 7168, 4),
    (1, 512, 5120, 4),
    (1, 512, 4096, 4),
    (1, 8192, 7168, 4),
    (1, 8192, 5120, 4),
    (1, 8192, 4096, 4),
]


@pytest.mark.benchmark
@pytest.mark.parametrize('n0,n1,h,mhc_mult', _POST_FWD_BENCH_CASES)
def test_mhc_post_fwd_benchmark(
    n0: int,
    n1: int,
    h: int,
    mhc_mult: int,
    benchmark_timer,
    benchmark_record,
) -> None:
    test_data = generate_mhc_post_test_data(n0=n0, n1=n1, h=h, mhc_mult=mhc_mult)
    x = test_data['x']
    residual = test_data['residual']
    post_layer_mix = test_data['post_layer_mix']
    comb_res_mix = test_data['comb_res_mix']
    out = torch.empty_like(residual)

    io_gb = count_bytes(comb_res_mix, residual, post_layer_mix, x, out) / 1e9

    t_us = benchmark_timer(lambda: mhc_post_fwd(x, residual, post_layer_mix, comb_res_mix, out))

    benchmark_record(
        kernel='mhc_post_fwd',
        operation='post_forward',
        params={'n0': n0, 'n1': n1, 'h': h, 'mhc_mult': mhc_mult},
        time_us=t_us,
        bandwidth_gbs=io_gb / t_us * 1e6,
        extras={'io_gb': io_gb},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('n0,n1,h,mhc_mult', _POST_FWD_BENCH_CASES)
def test_mhc_post_bwd_benchmark(
    n0: int,
    n1: int,
    h: int,
    mhc_mult: int,
    benchmark_timer,
    benchmark_record,
) -> None:
    test_data = generate_mhc_post_test_data(n0=n0, n1=n1, h=h, mhc_mult=mhc_mult)
    x = test_data['x']
    residual = test_data['residual']
    post_layer_mix = test_data['post_layer_mix']
    comb_res_mix = test_data['comb_res_mix']
    o_grad = test_data['o_grad']

    grad_x, grad_residual, grad_post_layer_mix, grad_comb_res_mix = mhc_post_bwd(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        o_grad,
    )
    io_gb = (
        count_bytes(
            o_grad,
            comb_res_mix,
            residual,
            post_layer_mix,
            x,
            grad_x,
            grad_residual,
            grad_post_layer_mix,
            grad_comb_res_mix,
        )
        / 1e9
    )

    t_us = benchmark_timer(lambda: mhc_post_bwd(x, residual, post_layer_mix, comb_res_mix, o_grad))

    benchmark_record(
        kernel='mhc_post_bwd',
        operation='post_backward',
        params={'n0': n0, 'n1': n1, 'h': h, 'mhc_mult': mhc_mult},
        time_us=t_us,
        bandwidth_gbs=io_gb / t_us * 1e6,
        extras={'io_gb': io_gb},
    )

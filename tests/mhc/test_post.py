from typing import Callable

import pytest
import torch
from tile_kernels.modeling.mhc.ops import mhc_post
from tile_kernels.torch.mhc import mhc_post_ref
from tests.conftest import IS_HIP

# mhc_post_kernel uses PDL (Programmatic Dependent Launch) which is SM90+ NV-only feature
pytestmark = pytest.mark.skipif(IS_HIP, reason='mhc_post_kernel uses PDL, an NV SM90+ feature not supported on HIP/AMD')


def generate_mhc_post_test_data(
    n0: int,
    n1: int,
    h: int,
    mhc_mult: int,
    device: str = 'cuda',
) -> dict[str, torch.Tensor]:
    x = torch.randn((n0, n1, h), dtype=torch.bfloat16, device=device)
    residual = torch.randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.randn((n0, n1, mhc_mult, 1), dtype=torch.float32, device=device)
    comb_res_mix = torch.randn((n0, n1, mhc_mult, mhc_mult), dtype=torch.float32, device=device)

    o_grad = torch.randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16, device=device)

    return {
        'x': x,
        'residual': residual,
        'post_layer_mix': post_layer_mix,
        'comb_res_mix': comb_res_mix,
        'o_grad': o_grad,
    }


def _tester(
    impl: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    test_data: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_ = test_data['x'].clone().requires_grad_()
    residual_ = test_data['residual'].clone().requires_grad_()
    post_layer_mix_ = test_data['post_layer_mix'].clone().requires_grad_()
    comb_res_mix_ = test_data['comb_res_mix'].clone().requires_grad_()
    out_ = impl(x_, residual_, post_layer_mix_, comb_res_mix_)
    torch.autograd.backward([out_], [test_data['o_grad']])
    return out_, x_.grad, residual_.grad, post_layer_mix_.grad, comb_res_mix_.grad


@pytest.mark.parametrize('n0', [1, 2])
@pytest.mark.parametrize('n1', [4096])
@pytest.mark.parametrize('h', [1280, 2560, 7168])
def test_mhc_post_comprehensive(n0: int, n1: int, h: int) -> None:
    test_data = generate_mhc_post_test_data(n0=n0, n1=n1, h=h, mhc_mult=4)

    out_tl, grad_x_tl, grad_residual_tl, grad_post_layer_mix_tl, grad_comb_res_mix_tl = _tester(
        mhc_post, test_data
    )
    out_ref, grad_x_ref, grad_residual_ref, grad_post_layer_mix_ref, grad_comb_res_mix_ref = _tester(
        mhc_post_ref, test_data
    )

    torch.testing.assert_close(out_tl, out_ref)
    torch.testing.assert_close(grad_x_tl, grad_x_ref)
    torch.testing.assert_close(grad_residual_tl, grad_residual_ref)
    torch.testing.assert_close(
        grad_post_layer_mix_tl,
        grad_post_layer_mix_ref,
        atol=1e-4,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        grad_comb_res_mix_tl,
        grad_comb_res_mix_ref,
        atol=1e-4,
        rtol=1e-4,
    )

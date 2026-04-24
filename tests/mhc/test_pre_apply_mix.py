from typing import Callable

import pytest
import torch
from tile_kernels.modeling.mhc.ops import mhc_pre_apply_mix
from tile_kernels.torch.mhc import mhc_pre_apply_mix_ref
from tests.conftest import IS_HIP

# mhc_pre_apply_mix kernel crashes on HIP/AMD (core dump, NV SM90+ feature dependency)
pytestmark = pytest.mark.skipif(IS_HIP, reason='mhc_pre_apply_mix kernel crashes on HIP/AMD targets (NV SM90+ dependency)')


def generate_pre_apply_mix_test_data(
    n0: int, n1: int, mhc: int, h: int, device: str = 'cuda'
) -> dict[str, torch.Tensor]:
    x = torch.randn(n0, n1, mhc, h, dtype=torch.bfloat16, device=device).sigmoid()
    mix = torch.randn(n0, n1, mhc, 1, dtype=torch.float32, device=device).softmax(-2)
    o_grad = torch.randn(n0, n1, h, dtype=torch.bfloat16, device=device)

    return {
        'x': x,
        'mix': mix,
        'o_grad': o_grad,
    }


def _tester(
    impl: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    test_data: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_ = test_data['x'].clone().requires_grad_()
    mix_ = test_data['mix'].clone().requires_grad_()
    o_ = impl(x_, mix_)
    x_.untyped_storage().grad_from_mhc_post = torch.zeros_like(x_)
    torch.autograd.backward([o_], [test_data['o_grad']])
    return (
        o_,
        x_.untyped_storage().grad_from_mhc_post if x_.grad is None else x_.grad,
        mix_.grad,
    )


@pytest.mark.parametrize('n0', [1, 2])
@pytest.mark.parametrize('n1', [1024, 4096])
@pytest.mark.parametrize('h', [1280, 2560, 7680])
def test_pre_apply_mix_comprehensive(n0: int, n1: int, h: int) -> None:
    mhc = 4

    test_data = generate_pre_apply_mix_test_data(n0=n0, n1=n1, mhc=mhc, h=h)

    o_tl, x_grad_tl, mix_grad_tl = _tester(mhc_pre_apply_mix, test_data)
    o_ref, x_grad_ref, mix_grad_ref = _tester(mhc_pre_apply_mix_ref, test_data)

    torch.testing.assert_close(o_tl, o_ref, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(x_grad_tl, x_grad_ref, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(mix_grad_tl, mix_grad_ref, atol=1e-2, rtol=1e-3)

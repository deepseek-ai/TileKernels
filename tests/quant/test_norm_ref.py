import pytest
import torch

from tile_kernels.torch.norm import norm_backward_ref, norm_forward_ref


@pytest.mark.parametrize('shape', [(32,), (16, 32), (2, 8, 32), (2, 3, 4, 32)])
@pytest.mark.parametrize('use_weight', [True, False])
@pytest.mark.parametrize('has_residual', [True, False])
@pytest.mark.parametrize('out_scale', [1.0, 0.5])
def test_norm_backward_ref_multidimensional(shape, use_weight, has_residual, out_scale):
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.float32, requires_grad=True)
    w = torch.randn(shape[-1], dtype=torch.float32, requires_grad=True) if use_weight else None
    res = torch.randn(shape, dtype=torch.float32, requires_grad=True) if has_residual else None
    eps = 1e-5

    out, rstd, norm_in = norm_forward_ref(x, w, eps, residual=res, out_scale=out_scale)
    out_grad = torch.randn_like(out)
    res_grad = torch.randn_like(out) if has_residual else None

    loss = (out * out_grad).sum()
    if has_residual:
        loss = loss + (norm_in * res_grad).sum()
    loss.backward()

    x_grad, w_grad = norm_backward_ref(
        out_grad=out_grad,
        x=norm_in.detach(),
        weight=w.detach() if use_weight else None,
        rstd=rstd.detach(),
        residual_out_grad=res_grad,
        out_scale=out_scale,
    )

    assert x_grad.shape == x.shape
    torch.testing.assert_close(x_grad, x.grad, atol=1e-5, rtol=1e-5)

    if use_weight:
        assert w_grad is not None
        assert w_grad.shape == w.shape
        torch.testing.assert_close(w_grad, w.grad, atol=1e-5, rtol=1e-5)
    else:
        assert w_grad is None

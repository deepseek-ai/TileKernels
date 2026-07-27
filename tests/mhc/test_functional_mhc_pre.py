import pytest
import torch
from tile_kernels.modeling.mhc.functional import mhc_pre


def generate_mhc_pre_test_data(
    n1: int,
    mhc_mult: int,
    hidden_size: int,
    generate_norm_weight: bool,
    norm_eps: float = 1e-6,
    post_mult_value: float = 1.0,
    pre_eps: float = 1e-6,
    sinkhorn_eps: float = 1e-6,
    sinkhorn_repeat: int = 10,
    n_splits: int = 16,
) -> dict[str, torch.Tensor]:
    n0 = 1
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    mhc_hidden_size = mhc_mult * hidden_size
    device = 'cuda'

    residual = (
        torch.randn((n0, n1, mhc_mult, hidden_size), dtype=torch.float, device=device)
        .mul(1 + torch.arange(mhc_mult, device=device).mul(0.01).view(1, 1, -1, 1))
        .bfloat16()
    )

    fn = (
        torch.randn((mhc_mult3, mhc_mult, hidden_size), dtype=torch.float, device=device)
        * 1e-4
        * (1 + torch.arange(mhc_mult, device=device).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)

    scale = torch.randn((3,), dtype=torch.float, device=device) * 0.1
    base = torch.randn((mhc_mult3,), dtype=torch.float, device=device) * 0.1

    if generate_norm_weight:
        norm_weight = torch.randn((mhc_hidden_size,), dtype=torch.float, device=device) * 0.1 + 1.0
    else:
        norm_weight = None

    return {
        'residual': residual,
        'fn': fn,
        'scale': scale,
        'base': base,
        'norm_weight': norm_weight,
        'norm_eps': norm_eps,
        'post_mult_value': post_mult_value,
        'pre_eps': pre_eps,
        'sinkhorn_eps': sinkhorn_eps,
        'sinkhorn_repeat': sinkhorn_repeat,
        'n_splits': n_splits,
    }


@pytest.mark.parametrize('n1', [4096, 8192])
@pytest.mark.parametrize('hidden_size', [1280, 2560, 7168])
@pytest.mark.parametrize('generate_norm_weight', [False, True])
def test_correctness(
    n1: int,
    hidden_size: int,
    generate_norm_weight: bool,
) -> None:
    mhc_mult = 4

    test_data = generate_mhc_pre_test_data(
        n1=n1,
        mhc_mult=mhc_mult,
        hidden_size=hidden_size,
        generate_norm_weight=generate_norm_weight,
    )

    train_layer_input, (train_post_mix, train_comb_mix) = mhc_pre(
        test_data['residual'],
        test_data['fn'],
        test_data['scale'],
        test_data['base'],
        norm_weight=test_data['norm_weight'],
        norm_eps=test_data['norm_eps'],
        mhc_mult=mhc_mult,
        post_mult_value=test_data['post_mult_value'],
        pre_eps=test_data['pre_eps'],
        sinkhorn_eps=test_data['sinkhorn_eps'],
        sinkhorn_repeat=test_data['sinkhorn_repeat'],
        n_splits=test_data['n_splits'],
    )

    with torch.no_grad():
        eval_layer_input, (eval_post_mix, eval_comb_mix) = mhc_pre(
            test_data['residual'],
            test_data['fn'],
            test_data['scale'],
            test_data['base'],
            norm_weight=test_data['norm_weight'],
            norm_eps=test_data['norm_eps'],
            mhc_mult=mhc_mult,
            post_mult_value=test_data['post_mult_value'],
            pre_eps=test_data['pre_eps'],
            sinkhorn_eps=test_data['sinkhorn_eps'],
            sinkhorn_repeat=test_data['sinkhorn_repeat'],
            n_splits=test_data['n_splits'],
    )

    torch.testing.assert_close(train_layer_input, eval_layer_input, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(train_post_mix, eval_post_mix, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(train_comb_mix, eval_comb_mix, atol=1e-4, rtol=1e-4)

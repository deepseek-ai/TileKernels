import pytest
import torch
from tile_kernels.modeling.mhc.ops import (
    mhc_pre_apply_mix,
    mhc_pre_big_fuse,
    mhc_pre_norm_fn,
    mhc_pre_split_mixes,
    sinkhorn_normalize,
)
from tests.conftest import IS_HIP

# mhc_pre_big_fuse depends on mhc_pre_apply_mix and mhc_pre_norm_fn which crash/fail on HIP/AMD
pytestmark = pytest.mark.skipif(IS_HIP, reason='mhc_pre_big_fuse depends on kernels not supported on HIP/AMD targets')


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

    mhc_scale = torch.randn((3,), dtype=torch.float, device=device) * 0.1
    mhc_base = torch.randn((mhc_mult3,), dtype=torch.float, device=device) * 0.1

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


def big_fuse_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mhc_mult = residual.shape[-2]

    mixes = mhc_pre_norm_fn(
        residual,
        fn,
        None,
        rms_eps,
        fuse_grad_acc=False,
        n_splits=n_splits,
    )

    pre_mix, post_mix, comb_mix = mhc_pre_split_mixes(
        mixes,
        mhc_scale,
        mhc_base,
        mhc_mult,
        mhc_post_mult_value,
        mhc_pre_eps,
    )

    comb_mix = sinkhorn_normalize(comb_mix, repeat=sinkhorn_repeat, eps=mhc_sinkhorn_eps)

    layer_input = mhc_pre_apply_mix(residual, pre_mix)

    return post_mix, comb_mix, layer_input


@pytest.mark.parametrize('n1', [512, 1024, 2048, 8192])
@pytest.mark.parametrize('hidden_size', [1280, 2560, 4096])
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

    post_mix_fused, comb_mix_fused, layer_input_fused = mhc_pre_big_fuse(
        test_data['residual'],
        test_data['fn'],
        test_data['mhc_scale'],
        test_data['mhc_base'],
        rms_eps=test_data['rms_eps'],
        mhc_pre_eps=test_data['mhc_pre_eps'],
        mhc_sinkhorn_eps=test_data['mhc_sinkhorn_eps'],
        mhc_post_mult_value=test_data['mhc_post_mult_value'],
        sinkhorn_repeat=test_data['sinkhorn_repeat'],
        n_splits=test_data['n_splits'],
    )

    post_mix_ref, comb_mix_ref, layer_input_ref = big_fuse_reference(
        test_data['residual'],
        test_data['fn'],
        test_data['mhc_scale'],
        test_data['mhc_base'],
        test_data['rms_eps'],
        test_data['mhc_pre_eps'],
        test_data['mhc_sinkhorn_eps'],
        test_data['mhc_post_mult_value'],
        test_data['sinkhorn_repeat'],
        test_data['n_splits'],
    )

    assert torch.equal(post_mix_fused, post_mix_ref)
    assert torch.equal(comb_mix_fused, comb_mix_ref)
    assert torch.equal(layer_input_fused, layer_input_ref)

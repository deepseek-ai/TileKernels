"""Correctness test: compare CUDA MHC kernels against PyTorch reference."""

import sys
import os
import math
import torch

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load PyTorch reference
from tile_kernels.torch.mhc import (
    expand_to_mhc_ref,
    mhc_pre_norm_fn_ref,
    mhc_pre_split_mixes_ref,
    sinkhorn_normalize_ref,
    mhc_pre_apply_mix_ref,
    mhc_post_ref,
    mhc_head_compute_mix_ref,
)

# Load CUDA extension
from tile_kernels.cuda_mhc.python.mhc_ops import (
    mhc_expand_cuda,
    mhc_pre_fused_cuda,
    mhc_post_cuda,
    mhc_head_cuda,
)


def compare(name, a, b, atol=1e-2):
    a_f = a.float().cpu()
    b_f = b.float().cpu()
    diff = (a_f - b_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    match = torch.allclose(a_f, b_f, atol=atol, rtol=1e-3)
    status = "PASS" if match else "FAIL"
    print(f"  [{status}] {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    return match


def test_expand():
    print("\n=== test_expand ===")
    H = 1280
    for M in [1, 8, 64]:
        x = torch.randn(M, H, dtype=torch.bfloat16, device='cuda')
        ref = expand_to_mhc_ref(x, 4)
        out = mhc_expand_cuda(x)
        compare(f"expand M={M}", ref, out, atol=0)


def test_post():
    print("\n=== test_post ===")
    H = 1280
    for M in [1, 8, 64]:
        x = torch.randn(M, H, dtype=torch.bfloat16, device='cuda')
        residual = torch.randn(M, 4, H, dtype=torch.bfloat16, device='cuda')
        post_mix_2d = torch.randn(M, 4, dtype=torch.float32, device='cuda')
        comb_mix_2d = torch.randn(M, 16, dtype=torch.float32, device='cuda')

        # Ref expects 4D: [batch=1, tokens=M, mhc, ...]
        ref_post = post_mix_2d.unsqueeze(0).unsqueeze(-1)  # [1, M, 4, 1]
        ref_comb = comb_mix_2d.view(1, M, 4, 4)            # [1, M, 4, 4]
        ref_residual = residual.unsqueeze(0)                 # [1, M, 4, H]
        ref_x = x.unsqueeze(0)                               # [1, M, H]
        ref = mhc_post_ref(ref_x, ref_residual, ref_post, ref_comb)
        ref = ref.squeeze(0)  # [M, 4, H]

        out = mhc_post_cuda(x, residual, post_mix_2d, comb_mix_2d)
        compare(f"post M={M}", ref, out, atol=1e-2)


def test_pre_fused():
    print("\n=== test_pre_fused ===")
    H = 1280
    mhc_mult = 4
    mhc_hidden = mhc_mult * H
    mhc_mult3 = mhc_mult * (2 + mhc_mult)

    for M in [1, 8]:
        torch.manual_seed(42)
        residual = torch.randn(M, mhc_mult, H, dtype=torch.bfloat16, device='cuda')
        fn = torch.randn(mhc_mult3, mhc_hidden, dtype=torch.float32, device='cuda')
        scale = torch.randn(3, dtype=torch.float32, device='cuda')
        base = torch.randn(mhc_mult3, dtype=torch.float32, device='cuda')

        rms_eps = 1e-6
        pre_eps = 1e-6
        sinkhorn_eps = 1e-6
        post_mult_value = 1.0
        sinkhorn_repeat = 10

        # CUDA kernel
        layer_input, post_mix, comb_mix = mhc_pre_fused_cuda(
            residual.flatten(1, 2), fn, scale, base,
            rms_eps, pre_eps, sinkhorn_eps, post_mult_value, sinkhorn_repeat)

        # PyTorch reference
        # mhc_pre_norm_fn_ref expects [batch, seq, mhc, H]
        residual_4d = residual.unsqueeze(0)  # [1, M, 4, H]
        norm_weight = None
        ref_mixes = mhc_pre_norm_fn_ref(residual_4d, fn, norm_weight, rms_eps)
        # ref_mixes: [1, M, 24]
        ref_pre, ref_post, ref_comb = mhc_pre_split_mixes_ref(
            ref_mixes, scale, base, mhc_mult, post_mult_value, pre_eps)
        # ref_pre: [1, M, 4, 1], ref_post: [1, M, 4, 1], ref_comb: [1, M, 4, 4]
        ref_comb = sinkhorn_normalize_ref(ref_comb, repeat=sinkhorn_repeat, eps=sinkhorn_eps)
        ref_layer_input = mhc_pre_apply_mix_ref(residual_4d, ref_pre)
        # ref_layer_input: [1, M, H]

        compare(f"pre_fused layer_input M={M}",
                ref_layer_input.squeeze(0), layer_input, atol=1e-1)
        compare(f"pre_fused post_mix M={M}",
                ref_post.squeeze(0).squeeze(-1), post_mix, atol=1e-1)
        compare(f"pre_fused comb_mix M={M}",
                ref_comb.squeeze(0).flatten(2), comb_mix.view(M, 4, 4).flatten(2), atol=1e-1)


def test_head():
    print("\n=== test_head ===")
    H = 1280
    mhc_mult = 4
    mhc_hidden = mhc_mult * H
    mhc_mult3 = mhc_mult * (2 + mhc_mult)

    for M in [1, 8]:
        torch.manual_seed(42)
        residual = torch.randn(M, mhc_mult, H, dtype=torch.bfloat16, device='cuda')
        fn = torch.randn(mhc_mult3, mhc_hidden, dtype=torch.float32, device='cuda')
        scale = torch.randn(1, dtype=torch.float32, device='cuda')
        base = torch.randn(mhc_mult, dtype=torch.float32, device='cuda')

        rms_eps = 1e-6
        pre_eps = 1e-6

        # CUDA kernel
        layer_input = mhc_head_cuda(
            residual.flatten(1, 2), fn, scale, base, rms_eps, pre_eps)

        # PyTorch reference
        import torch.nn.functional as F
        fn_padded = F.pad(fn, (0, 0, 0, mhc_mult3 - fn.shape[0])) if fn.shape[0] < mhc_mult3 else fn
        residual_4d = residual.unsqueeze(0)  # [1, M, 4, H]
        ref_mixes = mhc_pre_norm_fn_ref(residual_4d, fn_padded, None, rms_eps)
        ref_mixes = ref_mixes[..., :mhc_mult]  # [1, M, 4]
        ref_mix_val = mhc_head_compute_mix_ref(ref_mixes, scale, base, pre_eps)
        ref_layer_input = mhc_pre_apply_mix_ref(residual_4d, ref_mix_val.unsqueeze(-1))
        ref_layer_input = ref_layer_input.squeeze(0)  # [M, H]

        compare(f"head layer_input M={M}", ref_layer_input, layer_input, atol=1e-1)


if __name__ == '__main__':
    test_expand()
    test_post()
    test_pre_fused()
    test_head()
    print("\nDone!")

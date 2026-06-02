"""Benchmark CUDA MHC kernels vs PyTorch reference and TileLang."""

import sys
import os
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tile_kernels.cuda_mhc.python.mhc_ops import (
    mhc_expand_cuda,
    mhc_pre_fused_cuda,
    mhc_post_cuda,
    mhc_head_cuda,
)

# PyTorch reference implementations
from tile_kernels.torch.mhc import (
    expand_to_mhc_ref,
    mhc_pre_norm_fn_ref,
    mhc_pre_split_mixes_ref,
    sinkhorn_normalize_ref,
    mhc_pre_apply_mix_ref,
    mhc_post_ref,
    mhc_head_compute_mix_ref,
)
import torch.nn.functional as F


def benchmark(fn, *args, warmup=10, repeats=100, desc=""):
    """Benchmark a function, return median time in microseconds."""
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # microseconds

    times.sort()
    median = times[len(times) // 2]
    return median


def pytorch_pre_fused(residual_flat, fn, scale, base, rms_eps, pre_eps, sinkhorn_eps, post_mult_value, sinkhorn_repeat):
    """PyTorch reference for the full pre_fused pipeline."""
    res_flat = residual_flat.float()
    K = res_flat.shape[-1]
    out = res_flat @ fn.T
    sqrsum = res_flat.square().sum(-1, keepdim=True)
    rms = torch.rsqrt(sqrsum / K + rms_eps)
    out_normed = out * rms

    mhc_mult = 4
    pre = torch.sigmoid(out_normed[:, :mhc_mult]) + pre_eps
    post = torch.sigmoid(out_normed[:, mhc_mult:2*mhc_mult]) * post_mult_value

    comb = out_normed[:, 2*mhc_mult:]
    # Sinkhorn
    comb = comb.softmax(-1) + sinkhorn_eps
    comb = comb / (comb.sum(-2, keepdim=True) + sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + sinkhorn_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + sinkhorn_eps)

    # Apply mix
    M = residual_flat.shape[0]
    H = K // mhc_mult
    residual_3d = residual_flat.view(M, mhc_mult, H).float()
    layer_input = (residual_3d * pre.unsqueeze(-1)).sum(1).bfloat16()

    return layer_input, post, comb


def run_benchmarks():
    H = 1280
    mhc_mult = 4
    K = mhc_mult * H
    mhc_mult3 = 24
    rms_eps = 1e-6
    pre_eps = 1e-6
    sinkhorn_eps = 1e-6
    post_mult_value = 1.0
    sinkhorn_repeat = 10

    print("=" * 80)
    print(f"  CUDA MHC Benchmark — H={H}, mhc_mult={mhc_mult}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    # --- mhc_expand ---
    print("\n--- mhc_expand: [M, H] -> [M, 4, H] ---")
    print(f"  {'M':>6s}  {'CUDA (us)':>12s}  {'PyTorch (us)':>14s}  {'Speedup':>8s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*14}  {'-'*8}")

    for M in [1, 4, 16, 64, 256, 1024, 4096]:
        x = torch.randn(M, H, dtype=torch.bfloat16, device='cuda')

        cuda_t = benchmark(mhc_expand_cuda, x, desc=f"expand M={M}")
        pt_t = benchmark(expand_to_mhc_ref, x, mhc_mult, desc=f"expand_ref M={M}")

        print(f"  {M:>6d}  {cuda_t:>12.1f}  {pt_t:>14.1f}  {pt_t/cuda_t:>8.2f}x")

    # --- mhc_pre_fused ---
    print("\n--- mhc_pre_fused: GEMM+RMSNorm+Split+Sinkhorn+ApplyMix ---")
    print(f"  {'M':>6s}  {'CUDA (us)':>12s}  {'PyTorch (us)':>14s}  {'Speedup':>8s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*14}  {'-'*8}")

    for M in [1, 4, 16, 64, 256, 1024, 4096]:
        torch.manual_seed(42)
        residual = torch.randn(M, mhc_mult, H, dtype=torch.bfloat16, device='cuda')
        fn = torch.randn(mhc_mult3, K, dtype=torch.float32, device='cuda') * 0.1
        scale = torch.ones(3, dtype=torch.float32, device='cuda')
        base = torch.zeros(mhc_mult3, dtype=torch.float32, device='cuda')

        res_flat = residual.flatten(1, 2)

        cuda_t = benchmark(
            mhc_pre_fused_cuda,
            res_flat, fn, scale, base,
            rms_eps, pre_eps, sinkhorn_eps, post_mult_value, sinkhorn_repeat,
            desc=f"pre_fused M={M}")

        pt_t = benchmark(
            pytorch_pre_fused,
            res_flat, fn, scale, base,
            rms_eps, pre_eps, sinkhorn_eps, post_mult_value, sinkhorn_repeat,
            desc=f"pre_fused_ref M={M}")

        print(f"  {M:>6d}  {cuda_t:>12.1f}  {pt_t:>14.1f}  {pt_t/cuda_t:>8.2f}x")

    # --- mhc_post ---
    print("\n--- mhc_post: residual update ---")
    print(f"  {'M':>6s}  {'CUDA (us)':>12s}  {'PyTorch (us)':>14s}  {'Speedup':>8s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*14}  {'-'*8}")

    for M in [1, 4, 16, 64, 256, 1024, 4096]:
        x = torch.randn(M, H, dtype=torch.bfloat16, device='cuda')
        residual = torch.randn(M, mhc_mult, H, dtype=torch.bfloat16, device='cuda')
        post_mix = torch.randn(M, mhc_mult, dtype=torch.float32, device='cuda')
        comb_mix = torch.randn(M, mhc_mult * mhc_mult, dtype=torch.float32, device='cuda')

        cuda_t = benchmark(mhc_post_cuda, x, residual, post_mix, comb_mix, desc=f"post M={M}")

        # PyTorch reference
        def pt_post():
            residual_4d = residual.unsqueeze(0)
            x_4d = x.unsqueeze(0)
            post_4d = post_mix.unsqueeze(0).unsqueeze(-1)
            comb_4d = comb_mix.view(1, M, mhc_mult, mhc_mult)
            mhc_post_ref(x_4d, residual_4d, post_4d, comb_4d)

        pt_t = benchmark(pt_post, desc=f"post_ref M={M}")

        print(f"  {M:>6d}  {cuda_t:>12.1f}  {pt_t:>14.1f}  {pt_t/cuda_t:>8.2f}x")

    # --- mhc_head ---
    print("\n--- mhc_head: LM head preprocessing ---")
    print(f"  {'M':>6s}  {'CUDA (us)':>12s}  {'PyTorch (us)':>14s}  {'Speedup':>8s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*14}  {'-'*8}")

    for M in [1, 4, 16, 64, 256, 1024, 4096]:
        torch.manual_seed(42)
        residual = torch.randn(M, mhc_mult, H, dtype=torch.bfloat16, device='cuda')
        fn = torch.randn(mhc_mult3, K, dtype=torch.float32, device='cuda') * 0.1
        scale = torch.ones(1, dtype=torch.float32, device='cuda')
        base = torch.zeros(mhc_mult, dtype=torch.float32, device='cuda')

        res_flat = residual.flatten(1, 2)

        cuda_t = benchmark(mhc_head_cuda, res_flat, fn, scale, base, rms_eps, pre_eps, desc=f"head M={M}")

        def pt_head():
            fn_pad = F.pad(fn, (0, 0, 0, mhc_mult3 - fn.shape[0]))
            res_4d = residual.unsqueeze(0)
            mixes = mhc_pre_norm_fn_ref(res_4d, fn_pad, None, rms_eps)
            mixes = mixes[..., :mhc_mult]
            mix = mhc_head_compute_mix_ref(mixes, scale, base, pre_eps)
            mhc_pre_apply_mix_ref(res_4d, mix.unsqueeze(-1))

        pt_t = benchmark(pt_head, desc=f"head_ref M={M}")

        print(f"  {M:>6d}  {cuda_t:>12.1f}  {pt_t:>14.1f}  {pt_t/cuda_t:>8.2f}x")

    print("\n" + "=" * 80)
    print("  Benchmark complete.")
    print("=" * 80)


if __name__ == '__main__':
    run_benchmarks()

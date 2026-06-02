#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Expand embedding [M, H] -> [M, 4, H]
void mhc_expand_launch(
    const __nv_bfloat16* input,   // [M, H]
    __nv_bfloat16* output,        // [M, 4, H]
    int M, int H,
    cudaStream_t stream);

// Fused pre-processing: GEMM + RMSNorm + Split + Sinkhorn + ApplyMix
void mhc_pre_fused_launch(
    const __nv_bfloat16* residual, // [M, 4, H]
    const float* fn,               // [24, 4*H]
    const float* scale,            // [3]
    const float* base,             // [24]
    __nv_bfloat16* layer_input,    // [M, H] output
    float* post_mix,               // [M, 4] output
    float* comb_mix,               // [M, 16] output
    int M, int H,
    float rms_eps, float pre_eps, float sinkhorn_eps,
    float post_mult_value, int sinkhorn_repeat,
    cudaStream_t stream);

// Residual update: new_residual = x * post_mix + comb @ old_residual
void mhc_post_launch(
    const __nv_bfloat16* x,        // [M, H]
    const __nv_bfloat16* residual, // [M, 4, H]
    const float* post_mix,         // [M, 4]
    const float* comb_mix,         // [M, 16]
    __nv_bfloat16* output,         // [M, 4, H]
    int M, int H,
    cudaStream_t stream);

// LM head: thin GEMM (N=4) + RMSNorm + sigmoid + apply_mix
void mhc_head_launch(
    const __nv_bfloat16* residual, // [M, 4, H]
    const float* fn,               // [24, 4*H] (only first 4 rows used)
    const float* scale,            // [1]
    const float* base,             // [4]
    __nv_bfloat16* layer_input,    // [M, H] output
    int M, int H,
    float rms_eps, float pre_eps,
    cudaStream_t stream);

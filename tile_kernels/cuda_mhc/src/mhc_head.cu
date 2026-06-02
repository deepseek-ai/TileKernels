#include "mhc_common.cuh"

// mhc_head: thin GEMM (N=4) + RMSNorm + sigmoid + apply_mix
// Similar to mhc_pre_fused but N=4, no Sinkhorn, no post/comb

__global__ void mhc_head_kernel(
    const __nv_bfloat16* __restrict__ residual,  // [M, 4*H] flattened
    const float* __restrict__ fn,                // [24, 4*H] f32 (only first 4 rows used)
    const float* __restrict__ scale,             // [1] f32
    const float* __restrict__ base,              // [4] f32
    __nv_bfloat16* __restrict__ layer_input,     // [M, H] bf16
    int M, int H,
    float rms_eps, float pre_eps) {

    const int token = blockIdx.x;
    if (token >= M) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int K = MHC_MULT * H;

    // GEMM: [1, K] x [4, K]^T -> [4] + sqrsum
    float out[MHC_MULT] = {0};
    float sqrsum = 0.0f;

    const __nv_bfloat16* x_ptr = residual + token * K;
    const int K_TILE = 256;
    const int k_steps = K / K_TILE;

    for (int ks = 0; ks < k_steps; ks++) {
        int k_base = ks * K_TILE;
        for (int k = tid; k < K_TILE; k += nthreads) {
            float x_val = bf16_to_float(x_ptr[k_base + k]);
            sqrsum += x_val * x_val;
            for (int n = 0; n < MHC_MULT; n++) {
                out[n] += x_val * fn[n * K + k_base + k];
            }
        }
    }
    int k_done = k_steps * K_TILE;
    for (int k = k_done + tid; k < K; k += nthreads) {
        float x_val = bf16_to_float(x_ptr[k]);
        sqrsum += x_val * x_val;
        for (int n = 0; n < MHC_MULT; n++) {
            out[n] += x_val * fn[n * K + k];
        }
    }

    // Block-level reduction using shared memory
    __shared__ float s_partial[4];
    __shared__ float s_partial_out[4][MHC_MULT];
    __shared__ float s_sqrsum;
    __shared__ float s_out[MHC_MULT];
    __shared__ float s_mix[MHC_MULT];

    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int nwarps = nthreads / 32;

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sqrsum += __shfl_down_sync(0xffffffff, sqrsum, offset);
    }
    if (lane_id == 0) s_partial[warp_id] = sqrsum;

    for (int n = 0; n < MHC_MULT; n++) {
        float val = out[n];
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) s_partial_out[warp_id][n] = val;
    }
    __syncthreads();

    // Combine warp results - only first nwarps threads participate
    if (tid < nwarps) {
        float my_sqrsum = s_partial[tid];
        float my_out[MHC_MULT];
        for (int n = 0; n < MHC_MULT; n++) my_out[n] = s_partial_out[tid][n];
        unsigned mask = (1u << nwarps) - 1;
        for (int offset = nwarps / 2; offset > 0; offset >>= 1) {
            my_sqrsum += __shfl_down_sync(mask, my_sqrsum, offset);
            for (int n = 0; n < MHC_MULT; n++) {
                my_out[n] += __shfl_down_sync(mask, my_out[n], offset);
            }
        }
        if (tid == 0) {
            s_sqrsum = my_sqrsum;
            for (int n = 0; n < MHC_MULT; n++) s_out[n] = my_out[n];
        }
    }
    __syncthreads();

    if (tid == 0) {
        // RMSNorm
        float rms = fast_rsqrt(s_sqrsum / K + rms_eps);
        for (int n = 0; n < MHC_MULT; n++) {
            s_out[n] *= rms;
        }
        // Sigmoid
        for (int n = 0; n < MHC_MULT; n++) {
            s_mix[n] = fast_sigmoid(s_out[n] * scale[0] + base[n]) + pre_eps;
        }
    }
    __syncthreads();

    float my_mix[MHC_MULT];
    for (int j = 0; j < MHC_MULT; j++) {
        my_mix[j] = s_mix[j];
    }

    // ApplyMix
    const int H_TILE = 256;
    for (int h_base = 0; h_base < H; h_base += H_TILE) {
        int h_tile = min(H_TILE, H - h_base);
        for (int h_local = tid; h_local < h_tile; h_local += nthreads) {
            int h = h_base + h_local;
            float acc = 0.0f;
            for (int m = 0; m < MHC_MULT; m++) {
                float res = bf16_to_float(residual[token * MHC_MULT * H + m * H + h]);
                acc += my_mix[m] * res;
            }
            layer_input[token * H + h] = float_to_bf16(acc);
        }
    }
}

void mhc_head_launch(
    const __nv_bfloat16* residual,
    const float* fn,
    const float* scale,
    const float* base,
    __nv_bfloat16* layer_input,
    int M, int H,
    float rms_eps, float pre_eps,
    cudaStream_t stream) {

    const int threads = 128;
    mhc_head_kernel<<<M, threads, 0, stream>>>(
        residual, fn, scale, base, layer_input, M, H, rms_eps, pre_eps);
}

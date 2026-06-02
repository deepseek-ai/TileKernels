#include "mhc_common.cuh"

// ============================================================================
// mhc_pre_fused: GEMM + RMSNorm + Split + Sinkhorn + ApplyMix
//
// One thread block per token. 128 threads cooperate on the full GEMM.
// Block-level reduction via warp shuffle + shared memory combine.
//
// Performance (H20, H=1280):
//   M=1:   122 us    M=8:   127 us    M=64:  128 us
//   M=256: 135 us    M=1024: 288 us   M=4096: 994 us
// ============================================================================

template <int K_TILE, int H_TILE>
__global__ void mhc_pre_fused_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const float* __restrict__ fn,
    const float* __restrict__ scale,
    const float* __restrict__ base,
    __nv_bfloat16* __restrict__ layer_input,
    float* __restrict__ post_mix_out,
    float* __restrict__ comb_mix_out,
    int M, int H,
    float rms_eps, float pre_eps, float sinkhorn_eps,
    float post_mult_value, int sinkhorn_repeat) {

    const int token = blockIdx.x;
    if (token >= M) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int K = MHC_MULT * H;

    float out[MHC_MULT3] = {0};
    float sqrsum = 0.0f;

    const __nv_bfloat16* x_ptr = residual + token * K;
    const int k_steps = K / K_TILE;

    for (int ks = 0; ks < k_steps; ks++) {
        int k_base = ks * K_TILE;
        for (int k = tid; k < K_TILE; k += nthreads) {
            float x_val = bf16_to_float(x_ptr[k_base + k]);
            sqrsum += x_val * x_val;
            for (int n = 0; n < MHC_MULT3; n++) {
                out[n] += x_val * fn[n * K + k_base + k];
            }
        }
    }
    int k_done = k_steps * K_TILE;
    for (int k = k_done + tid; k < K; k += nthreads) {
        float x_val = bf16_to_float(x_ptr[k]);
        sqrsum += x_val * x_val;
        for (int n = 0; n < MHC_MULT3; n++) {
            out[n] += x_val * fn[n * K + k];
        }
    }

    // Block-level reduction
    __shared__ float s_partial[4];
    __shared__ float s_partial_out[4][MHC_MULT3];

    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int nwarps = nthreads / 32;

    for (int offset = 16; offset > 0; offset >>= 1) {
        sqrsum += __shfl_down_sync(0xffffffff, sqrsum, offset);
    }
    if (lane_id == 0) s_partial[warp_id] = sqrsum;

    for (int n = 0; n < MHC_MULT3; n++) {
        float val = out[n];
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) s_partial_out[warp_id][n] = val;
    }
    __syncthreads();

    __shared__ float s_sqrsum;
    __shared__ float s_out[MHC_MULT3];
    __shared__ float s_pre[MHC_MULT];

    if (tid < nwarps) {
        float my_sqrsum = s_partial[tid];
        float my_out[MHC_MULT3];
        for (int n = 0; n < MHC_MULT3; n++) my_out[n] = s_partial_out[tid][n];
        unsigned mask = (1u << nwarps) - 1;
        for (int offset = nwarps / 2; offset > 0; offset >>= 1) {
            my_sqrsum += __shfl_down_sync(mask, my_sqrsum, offset);
            for (int n = 0; n < MHC_MULT3; n++) {
                my_out[n] += __shfl_down_sync(mask, my_out[n], offset);
            }
        }
        if (tid == 0) {
            s_sqrsum = my_sqrsum;
            for (int n = 0; n < MHC_MULT3; n++) s_out[n] = my_out[n];
        }
    }
    __syncthreads();

    if (tid == 0) {
        float rms = fast_rsqrt(s_sqrsum / K + rms_eps);
        for (int n = 0; n < MHC_MULT3; n++) s_out[n] *= rms;

        for (int j = 0; j < MHC_MULT; j++) {
            s_pre[j] = fast_sigmoid(s_out[j] * scale[0] + base[j]) + pre_eps;
        }
        for (int j = 0; j < MHC_MULT; j++) {
            post_mix_out[token * MHC_MULT + j] =
                fast_sigmoid(s_out[MHC_MULT + j] * scale[1] + base[MHC_MULT + j]) * post_mult_value;
        }

        float cm[MHC_MULT][MHC_MULT];
        for (int j = 0; j < MHC_MULT; j++)
            for (int k = 0; k < MHC_MULT; k++)
                cm[j][k] = s_out[MHC_MULT * 2 + j * MHC_MULT + k] * scale[2]
                          + base[MHC_MULT * 2 + j * MHC_MULT + k];

        float row_max[MHC_MULT], row_sum[MHC_MULT], col_sum[MHC_MULT];
        for (int j = 0; j < MHC_MULT; j++) {
            row_max[j] = -1e30f;
            for (int k = 0; k < MHC_MULT; k++) row_max[j] = fmaxf(row_max[j], cm[j][k]);
        }
        for (int j = 0; j < MHC_MULT; j++) {
            row_sum[j] = 0.0f;
            for (int k = 0; k < MHC_MULT; k++) {
                cm[j][k] = expf(cm[j][k] - row_max[j]);
                row_sum[j] += cm[j][k];
            }
        }
        for (int j = 0; j < MHC_MULT; j++)
            for (int k = 0; k < MHC_MULT; k++)
                cm[j][k] = cm[j][k] / row_sum[j] + sinkhorn_eps;

        for (int k = 0; k < MHC_MULT; k++) {
            col_sum[k] = 0.0f;
            for (int j = 0; j < MHC_MULT; j++) col_sum[k] += cm[j][k];
        }
        for (int j = 0; j < MHC_MULT; j++)
            for (int k = 0; k < MHC_MULT; k++)
                cm[j][k] = cm[j][k] / (col_sum[k] + sinkhorn_eps);

        for (int iter = 1; iter < sinkhorn_repeat; iter++) {
            for (int j = 0; j < MHC_MULT; j++) {
                row_sum[j] = 0.0f;
                for (int k = 0; k < MHC_MULT; k++) row_sum[j] += cm[j][k];
            }
            for (int j = 0; j < MHC_MULT; j++)
                for (int k = 0; k < MHC_MULT; k++)
                    cm[j][k] = cm[j][k] / (row_sum[j] + sinkhorn_eps);
            for (int k = 0; k < MHC_MULT; k++) {
                col_sum[k] = 0.0f;
                for (int j = 0; j < MHC_MULT; j++) col_sum[k] += cm[j][k];
            }
            for (int j = 0; j < MHC_MULT; j++)
                for (int k = 0; k < MHC_MULT; k++)
                    cm[j][k] = cm[j][k] / (col_sum[k] + sinkhorn_eps);
        }

        for (int j = 0; j < MHC_MULT; j++)
            for (int k = 0; k < MHC_MULT; k++)
                comb_mix_out[token * MHC_MULT2 + j * MHC_MULT + k] = cm[j][k];
    }
    __syncthreads();

    float my_pre[MHC_MULT];
    for (int j = 0; j < MHC_MULT; j++) my_pre[j] = s_pre[j];

    for (int h_base = 0; h_base < H; h_base += H_TILE) {
        int h_tile = min(H_TILE, H - h_base);
        for (int h_local = tid; h_local < h_tile; h_local += nthreads) {
            int h = h_base + h_local;
            float acc = 0.0f;
            for (int m = 0; m < MHC_MULT; m++) {
                float res = bf16_to_float(residual[token * MHC_MULT * H + m * H + h]);
                acc += my_pre[m] * res;
            }
            layer_input[token * H + h] = float_to_bf16(acc);
        }
    }
}

void mhc_pre_fused_launch(
    const __nv_bfloat16* residual,
    const float* fn,
    const float* scale,
    const float* base,
    __nv_bfloat16* layer_input,
    float* post_mix,
    float* comb_mix,
    int M, int H,
    float rms_eps, float pre_eps, float sinkhorn_eps,
    float post_mult_value, int sinkhorn_repeat,
    cudaStream_t stream) {

    constexpr int threads = 128;
    constexpr int K_TILE = 256;
    constexpr int H_TILE = 256;
    mhc_pre_fused_kernel<K_TILE, H_TILE>
        <<<M, threads, 0, stream>>>(
            residual, fn, scale, base, layer_input,
            post_mix, comb_mix,
            M, H, rms_eps, pre_eps, sinkhorn_eps,
            post_mult_value, sinkhorn_repeat);
}

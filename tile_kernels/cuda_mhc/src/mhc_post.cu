#include "mhc_common.cuh"

// mhc_post: new_residual[m', h] = post_mix[m'] * x[h] + sum_m(comb[m, m'] * residual[m, h])
// Grid: M blocks, each block handles 1 token
// Block: 256 threads, iterate over H in tiles

template <int H_TILE>
__global__ void mhc_post_kernel(
    const __nv_bfloat16* __restrict__ x,        // [M, H]
    const __nv_bfloat16* __restrict__ residual,  // [M, 4, H]
    const float* __restrict__ post_mix,          // [M, 4]
    const float* __restrict__ comb_mix,          // [M, 16]
    __nv_bfloat16* __restrict__ output,          // [M, 4, H]
    int M, int H) {

    const int token = blockIdx.x;
    if (token >= M) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // Load comb[4,4] into registers (16 floats, all threads load the same)
    float comb[MHC_MULT][MHC_MULT];
    const float* comb_ptr = comb_mix + token * MHC_MULT2;
    for (int j = 0; j < MHC_MULT; j++) {
        for (int k = 0; k < MHC_MULT; k++) {
            // Use first warp to load, broadcast not needed since all threads read same data
            comb[j][k] = comb_ptr[j * MHC_MULT + k];
        }
    }

    // Load post_mix[4] into registers
    float pm[MHC_MULT];
    const float* pm_ptr = post_mix + token * MHC_MULT;
    for (int j = 0; j < MHC_MULT; j++) {
        pm[j] = pm_ptr[j];
    }

    // Iterate over H in tiles
    for (int h_base = 0; h_base < H; h_base += H_TILE) {
        const int h_tile = min(H_TILE, H - h_base);

        // Each thread handles a subset of the h_tile elements
        for (int h_local = tid; h_local < h_tile; h_local += nthreads) {
            const int h = h_base + h_local;

            // Load x[h]
            float x_val = bf16_to_float(x[token * H + h]);

            // Compute for each output head
            for (int m_out = 0; m_out < MHC_MULT; m_out++) {
                float acc = pm[m_out] * x_val;
                for (int m_in = 0; m_in < MHC_MULT; m_in++) {
                    float res = bf16_to_float(residual[token * MHC_MULT * H + m_in * H + h]);
                    acc += comb[m_in][m_out] * res;
                }
                output[token * MHC_MULT * H + m_out * H + h] = float_to_bf16(acc);
            }
        }
    }
}

void mhc_post_launch(
    const __nv_bfloat16* x,
    const __nv_bfloat16* residual,
    const float* post_mix,
    const float* comb_mix,
    __nv_bfloat16* output,
    int M, int H,
    cudaStream_t stream) {

    const int threads = 256;
    const int H_TILE = 256;  // Process 256 elements at a time
    mhc_post_kernel<H_TILE><<<M, threads, 0, stream>>>(x, residual, post_mix, comb_mix, output, M, H);
}

#include "mhc_common.cuh"

// mhc_expand: [M, H] -> [M, 4, H]
// Each thread copies 16 bytes (8 bf16 values) from input, writes 4 copies to output
// Grid: enough blocks to cover M * H / 8 elements, each block has 256 threads

__global__ void mhc_expand_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int M, int H) {

    // Total bf16 elements to process per "copy unit" = M * H
    // Each thread handles 8 bf16 values (one uint4 = 16 bytes)
    const int elems_per_thread = 8;
    const int total_units = M * (H / elems_per_thread);
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = tid; idx < total_units; idx += blockDim.x * gridDim.x) {
        int h_unit = idx % (H / elems_per_thread);  // which 8-element chunk in H
        int m = idx / (H / elems_per_thread);        // which token

        // Load 8 bf16 values (16 bytes) using uint4
        const uint4* src = reinterpret_cast<const uint4*>(
            input + m * H + h_unit * elems_per_thread);
        uint4 val = *src;

        // Write 4 copies to output[m, 0..3, h_unit*8 .. h_unit*8+7]
        for (int mc = 0; mc < MHC_MULT; mc++) {
            uint4* dst = reinterpret_cast<uint4*>(
                output + (m * MHC_MULT + mc) * H + h_unit * elems_per_thread);
            *dst = val;
        }
    }
}

void mhc_expand_launch(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int M, int H,
    cudaStream_t stream) {

    const int elems_per_thread = 8;
    const int H_units = H / elems_per_thread;
    const int total_units = M * H_units;
    const int threads = 256;
    const int blocks = (total_units + threads - 1) / threads;

    mhc_expand_kernel<<<blocks, threads, 0, stream>>>(input, output, M, H);
}

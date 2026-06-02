#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

constexpr int MHC_MULT   = 4;
constexpr int MHC_MULT2  = MHC_MULT * MHC_MULT;
constexpr int MHC_MULT3  = MHC_MULT * (2 + MHC_MULT);

__device__ __forceinline__ float fast_sigmoid(float x) {
    float neg_x = -x;
    float e = __expf(neg_x);
    return 1.0f / (1.0f + e);
}

__device__ __forceinline__ float fast_rsqrt(float x) {
    return __frsqrt_rn(x);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float f) {
    return __float2bfloat16(f);
}

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

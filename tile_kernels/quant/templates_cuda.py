PREFETCH_L2_SOURCE = """
__device__ __forceinline__ int tk_prefetch_l2(const void* addr) {
    asm volatile("prefetch.global.L2 [%0];" : : "l"(__cvta_generic_to_global(addr)) : "memory");
    return 0;
}
"""


# Requires a nonnegative limit. NaN propagates (min.NaN semantics).
SYMMETRIC_CLAMP_SOURCE = """
__device__ __forceinline__ float tk_symmetric_clamp(float x, float limit) {
    float result;
    asm("min.NaN.xorsign.abs.f32 %0, %1, %2;"
        : "=f"(result) : "f"(x), "f"(limit));
    return result;
}
"""


# Upper-bound clamp with NaN propagation (min.NaN vs min: the latter is NaN-absorbing).
UPPER_CLAMP_SOURCE = """
__device__ __forceinline__ float tk_upper_clamp(float x, float limit) {
    float result;
    asm("min.NaN.f32 %0, %1, %2;"
        : "=f"(result) : "f"(x), "f"(limit));
    return result;
}
"""


# Fast exp with FTZ; the multiplier is FP32 log2(e).
EXP_FTZ_SOURCE = """
__device__ __forceinline__ float tk_exp_ftz(float x) {
    float result;
    asm("ex2.approx.ftz.f32 %0, %1;"
        : "=f"(result) : "f"(x * 0x1.715476p+0f));
    return result;
}
"""

import tilelang
from tilelang import language as T

from tile_kernels.utils import align

# Hardware warp-wide reduction. Tilelang currently lacks the proper macro condition to
# enable redux.
_WARP_REDUX_SRC = r"""
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && \
    (defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL) || \
     defined(__CUDA_ARCH_FEAT_SM100_F) || defined(__CUDA_ARCH_FEAT_SM103_F))
#define TL_HAS_REDUX_F32 1
#endif
__device__ __forceinline__ float tl_redux_max_f32(float x) {
#ifdef TL_HAS_REDUX_F32
  float r;
  asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(r) : "f"(x));
  return r;
#else
  // Portable fallback: same exact operation, 5-step butterfly (identical result).
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 16));
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 8));
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 4));
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 2));
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 1));
  return x;
#endif
}
__device__ __forceinline__ int tl_redux_min_s32(int x) {
#ifdef TL_HAS_REDUX_F32
  int r;
  asm volatile("redux.sync.min.s32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(x));
  return r;
#else
  x = min(x, __shfl_xor_sync(0xffffffffu, x, 16));
  x = min(x, __shfl_xor_sync(0xffffffffu, x, 8));
  x = min(x, __shfl_xor_sync(0xffffffffu, x, 4));
  x = min(x, __shfl_xor_sync(0xffffffffu, x, 2));
  x = min(x, __shfl_xor_sync(0xffffffffu, x, 1));
  return x;
#endif
}
"""


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_topk_gate_kernel_cuda(num_experts: int, num_topk: int, use_pdl: bool = False):
    num_tokens = T.dynamic('num_tokens')
    num_threads = 32
    num_aligned_experts = align(num_experts, num_threads)

    @T.prim_func
    def topk_gate_kernel(
        scores: T.Tensor[(num_tokens, num_experts), T.float32],
        topk_idx: T.Tensor[(num_tokens, num_topk), T.int64],
    ):
        with T.Kernel(num_tokens, threads=num_threads) as pid:
            T.import_source(_WARP_REDUX_SRC)
            if use_pdl:
                T.pdl_sync()
            scores_fragment = T.alloc_fragment((num_aligned_experts,), T.float32)
            nonfinite_reducer = T.alloc_reducer((1,), T.int32, op='sum')
            nonfinite_result = T.alloc_fragment((1,), T.int32)
            amax_local = T.alloc_local((1,), T.float32)
            idx_result_local = T.alloc_local((1,), T.int32)

            for i in T.Parallel(num_aligned_experts):
                if i < num_experts:
                    scores_fragment[i] = scores[pid, i]
                else:
                    scores_fragment[i] = -T.infinity(T.float32)

            # Invalid input check, single deferred assert to keep the load loop vectorized
            T.reducer_init(nonfinite_reducer)
            for i in T.Parallel(num_aligned_experts):
                if i < num_experts and not T.isfinite(scores_fragment[i]):
                    T.reducer_update(nonfinite_reducer[0], 1)
            T.finalize_reducer(nonfinite_reducer, nonfinite_result)
            T.device_assert(nonfinite_result[0] == 0, msg='Score input for topk gate must be finite')

            # Get topk via repeatedly finding max
            for k in T.unroll(num_topk):
                amax_local[0] = -T.infinity(T.float32)
                for i in T.Parallel(num_aligned_experts):
                    amax_local[0] = T.max(amax_local[0], scores_fragment[i])
                amax_local[0] = T.call_extern(T.float32, 'tl_redux_max_f32', amax_local[0])
                idx_result_local[0] = 2147483647
                for i in T.Parallel(num_aligned_experts):
                    if scores_fragment[i] == amax_local[0]:
                        idx_result_local[0] = T.min(idx_result_local[0], i)
                idx_result_local[0] = T.call_extern(T.int32, 'tl_redux_min_s32', idx_result_local[0])
                if T.get_thread_binding() == 0:
                    topk_idx[pid, k] = T.Cast(T.int64, idx_result_local[0])
                for i in T.Parallel(num_aligned_experts):
                    if i == idx_result_local[0]:
                        scores_fragment[i] = -T.infinity(T.float32)

            if use_pdl:
                T.pdl_trigger()

    return topk_gate_kernel

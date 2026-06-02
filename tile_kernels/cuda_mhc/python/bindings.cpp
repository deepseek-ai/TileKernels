#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "mhc_kernels.cuh"

// ============================================================================
// mhc_expand: [M, H] -> [M, 4, H]
// ============================================================================
void mhc_expand_cuda(
    torch::Tensor input,    // [M, H] bf16
    torch::Tensor output) { // [M, 4, H] bf16
    TORCH_CHECK(input.dtype() == torch::kBFloat16);
    TORCH_CHECK(output.dtype() == torch::kBFloat16);
    TORCH_CHECK(input.is_contiguous());
    TORCH_CHECK(output.is_contiguous());

    int M = input.size(0);
    int H = input.size(1);
    TORCH_CHECK(output.size(0) == M);
    TORCH_CHECK(output.size(1) == 4);
    TORCH_CHECK(output.size(2) == H);

    mhc_expand_launch(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        M, H,
        c10::cuda::getCurrentCUDAStream());
}

// ============================================================================
// mhc_pre_fused: GEMM + RMSNorm + Split + Sinkhorn + ApplyMix
// ============================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mhc_pre_fused_cuda(
    torch::Tensor residual, // [M, 4, H] bf16 or [M, 4*H] bf16
    torch::Tensor fn,       // [24, 4*H] f32
    torch::Tensor scale,    // [3] f32
    torch::Tensor base,     // [24] f32
    double rms_eps,
    double pre_eps,
    double sinkhorn_eps,
    double post_mult_value,
    int64_t sinkhorn_repeat) {
    TORCH_CHECK(residual.dtype() == torch::kBFloat16);
    TORCH_CHECK(fn.dtype() == torch::kFloat32);
    TORCH_CHECK(residual.is_contiguous());
    TORCH_CHECK(fn.is_contiguous());

    int M = residual.size(0);
    int H = residual.dim() == 3 ? residual.size(2) : residual.size(1) / 4;

    auto layer_input = torch::empty({M, H}, residual.options());
    auto post_mix = torch::empty({M, 4}, torch::dtype(torch::kFloat32).device(residual.device()));
    auto comb_mix = torch::empty({M, 16}, torch::dtype(torch::kFloat32).device(residual.device()));

    mhc_pre_fused_launch(
        reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        fn.data_ptr<float>(),
        scale.data_ptr<float>(),
        base.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(layer_input.data_ptr<at::BFloat16>()),
        post_mix.data_ptr<float>(),
        comb_mix.data_ptr<float>(),
        M, H,
        static_cast<float>(rms_eps),
        static_cast<float>(pre_eps),
        static_cast<float>(sinkhorn_eps),
        static_cast<float>(post_mult_value),
        static_cast<int>(sinkhorn_repeat),
        c10::cuda::getCurrentCUDAStream());

    return {layer_input, post_mix, comb_mix};
}

// ============================================================================
// mhc_post: residual update
// ============================================================================
torch::Tensor mhc_post_cuda(
    torch::Tensor x,            // [M, H] bf16
    torch::Tensor residual,     // [M, 4, H] bf16 or [M, 4*H] bf16
    torch::Tensor post_mix,     // [M, 4] f32
    torch::Tensor comb_mix) {   // [M, 16] f32
    TORCH_CHECK(x.dtype() == torch::kBFloat16);
    TORCH_CHECK(residual.dtype() == torch::kBFloat16);
    TORCH_CHECK(post_mix.dtype() == torch::kFloat32);
    TORCH_CHECK(comb_mix.dtype() == torch::kFloat32);

    int M = residual.size(0);
    int H = residual.dim() == 3 ? residual.size(2) : residual.size(1) / 4;

    auto output = torch::empty_like(residual);

    mhc_post_launch(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        post_mix.data_ptr<float>(),
        comb_mix.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        M, H,
        c10::cuda::getCurrentCUDAStream());

    return output;
}

// ============================================================================
// mhc_head: LM head preprocessing
// ============================================================================
torch::Tensor mhc_head_cuda(
    torch::Tensor residual, // [M, 4, H] bf16 or [M, 4*H] bf16
    torch::Tensor fn,       // [24, 4*H] f32 (first 4 rows used)
    torch::Tensor scale,    // [1] f32
    torch::Tensor base,     // [4] f32
    double rms_eps,
    double pre_eps) {
    TORCH_CHECK(residual.dtype() == torch::kBFloat16);
    TORCH_CHECK(fn.dtype() == torch::kFloat32);

    int M = residual.size(0);
    int H = residual.dim() == 3 ? residual.size(2) : residual.size(1) / 4;

    auto layer_input = torch::empty({M, H}, residual.options());

    mhc_head_launch(
        reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        fn.data_ptr<float>(),
        scale.data_ptr<float>(),
        base.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(layer_input.data_ptr<at::BFloat16>()),
        M, H,
        static_cast<float>(rms_eps),
        static_cast<float>(pre_eps),
        c10::cuda::getCurrentCUDAStream());

    return layer_input;
}

// ============================================================================
// pybind11 module
// ============================================================================
PYBIND11_MODULE(_mhc_cuda, m) {
    m.def("mhc_expand_cuda", &mhc_expand_cuda,
          "Expand embedding [M,H] -> [M,4,H]");
    m.def("mhc_pre_fused_cuda", &mhc_pre_fused_cuda,
          "Fused MHC pre-processing");
    m.def("mhc_post_cuda", &mhc_post_cuda,
          "MHC residual update");
    m.def("mhc_head_cuda", &mhc_head_cuda,
          "MHC LM head preprocessing");
}

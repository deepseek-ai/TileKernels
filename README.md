# Tile Kernels

TileKernels is a library of dozens of highly optimized kernels implemented in [TileLang](https://github.com/tile-ai/tilelang), a domain-specific language supporting multiple hardware backends. It provides kernels for several common operations in LLM training and inference, including mixture-of-experts routing, Engram, quantization, and manifold hyper-connections. Most kernels achieve performance close to the hardware's compute or memory bandwidth limits. All of these kernels have already been used in our internal training and inference workloads.

> TileKernels 是一个包含数十个深度优化算子的高性能算子库，基于支持多种硬件后端的领域专用语言 [TileLang](https://github.com/tile-ai/tilelang) 实现。它为大语言模型训练与推理中的几项常见操作提供算子，包括混合专家路由、Engram、量化和流形超连接（mHC）。大多数算子的性能接近硬件的计算吞吐或内存带宽上限。全部算子已用于我们的内部训练与推理任务。

## News

- **[2026-09-30] Huawei Ascend support**: Added Huawei Ascend support and updated the usage documentation. Following the NVIDIA path, the kernels now ship a second backend that is selected automatically at runtime, so the same Python APIs run on both NVIDIA GPUs and Huawei NPUs.

## Features

- **MoE Routing** — Top-k expert selection and scoring for Mixture of Experts routing
- **Quantization** — Per-token, per-block, and per-channel FP8/FP4 casting and dequantization, with fused SwiGLU+quantization ops
- **Engram** — Engram gating kernels with fused RMSNorm, forward/backward passes and weight gradient reduction
- **Manifold HyperConnection** — Hyper-connection kernels including Sinkhorn normalization and mix splitting/application
- **Transform** — RoPE kernel
- **Rand** - Rand kernel
- **Modeling** — High-level `torch.autograd.Function` wrapper for the Engram gate

## Requirements

- Python 3.12 or higher
- PyTorch 2.13 or higher
- TileLang 0.1.15 or higher
- CUDA Backend
  - NVIDIA SM90 or SM100 architecture GPU
  - CUDA Toolkit 13.1 or higher
- Ascend Backend
  - Ascend 950 NPU
  - CANN 9.3.0 or higher

## Installation

### Install a local development version

```bash
pip install -e ".[dev]"
```

### Install a release version

```bash
pip install tile-kernels
```

## Testing

Tests using pytest:

### Test single test file

```bash
python -m pytest tests/quant/test_per_token_cast.py -n 4 # Correctness only with 4 workers
python -m pytest tests/quant/test_per_token_cast.py --run-benchmark # Correctness + Benchmarking
```

### Test level

```bash
TK_TEST_LEVEL=0 python -m pytest -n 4 # Core tests
TK_TEST_LEVEL=2 python -m pytest -n 4 --count 2 # Full tests
```

## Development

Install pre commit hooks.

```bash
pre-commit install --install-hooks
```

## Project Structure

```txt
tile_kernels/
├── engram/     # Engram gating kernels
├── mhc/        # Manifold HyperConnection kernels
├── modeling/   # High-level autograd modeling layer (engram)
├── moe/        # Mixture of Experts routing kernels
├── quant/      # Quantization kernels
├── rand/       # Random number generator kernel
├── testing/    # Test and benchmark utilities
├── torch/      # PyTorch reference implementations
└── transform/  # Rotary position embedding kernel
```

## Acknowledgement

This project is built on [TileLang](https://github.com/tile-ai/tilelang), and we extend our thanks and respect to its developers. We also gratefully acknowledge Huawei for its technical support and engineering expertise throughout the development of Tile Kernels' Ascend backend.

## License

This code repository is released under [the MIT License](LICENSE).

## Citation

```bibtex
@misc{tilekernels,
      title={TileKernels},
      author={Xiangwen Wang, Chenhao Xu, Huanqi Cao, Luotian Huang, Yuxuan Zhou, Weilin Zhao, Rui Tian, Anyi Xu, Fucong Dai, Kuai Yu, Ruifan Xu, Yi Qian, Shengyuan Jia, Chenggang Zhao, Chaofan Lin, Wei Zhang and Lei Wang},
      year={2026},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/TileKernels}},
}
```

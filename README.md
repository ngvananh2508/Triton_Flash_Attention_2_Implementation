# Custom Triton Flash-Attention 2 Implementation

This repository contains a high-performance, custom-built **FlashAttention-2** forward pass implementation using **Triton**. 

The project demonstrates how to bridge PyTorch tensors with low-level GPU kernels, focusing on memory efficiency and speed by utilizing the online softmax algorithm and register-level optimizations.

## Features
* **Custom Triton Kernel**: Built from scratch to compute attention scores without materializing the $N \times N$ attention matrix in VRAM.
* **Online Softmax**: Implements running max and sum metrics within registers, significantly reducing memory footprint to $O(N)$.
* **Autotuning**: Uses `@triton.autotune` to dynamically select the optimal block sizes (`BLOCK_Q`, `BLOCK_K`) for varying sequence lengths and GPU architectures.
* **Shared Memory Tiling**: Leverages tiling strategies to improve memory coalescing when accessing transposed Key ($K^T$) matrices.
* **Benchmark Suite**: Includes a comprehensive performance comparison against `torch.nn.functional.scaled_dot_product_attention` (SDPA) and a naive eager-mode implementation.

## Performance
The implementation achieves performance parity with PyTorch's native `SDPA` FlashAttention backend, often exceeding it at specific sequence lengths.

| Seq Length (N) | vs Naive PyTorch | vs Native SDPA |
| :--- | :--- | :--- |
| 1024 | 1.53x | 0.73x |
| 2048 | 1.96x | 2.05x |
| 4096 | 3.19x | 0.96x |
| 8192 | 4.56x | 1.14x |
| 16384 | 4.32x | 1.01x |

*(Results based on FP16 benchmarks on NVIDIA GPU hardware.)*

## Prerequisites
* Python 3.12+
* PyTorch (with CUDA support)
* Triton (`pip install triton`)

This implementation is inspired by the FlashAttention-2 algorithm (Dao et al.) and the Triton language documentation.
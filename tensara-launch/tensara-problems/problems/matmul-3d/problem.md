---
slug: "matmul-3d"
title: "3D Tensor-Matrix Multiplication"
difficulty: "HARD"
author: "sarthak"
tags: ["matmul"]
---

Perform 3D tensor-matrix multiplication of two tensors:
$$
C[i][j][l] = \sum_{k=0}^{K-1} A[i][j][k] \cdot B[k][l]
$$

## Input
- Tensor $A$ of size $N \times M \times K$
- Matrix $B$ of size $K \times L$

## Output
- Tensor $C$ of size $N \times M \times L$

## Notes:
- All tensors $\text{A}$, $\text{B}$, and $\text{C}$ are stored in row-major order
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/10_3D_tensor_matrix_multiplication.py)

## Test Case Sizes

- 32x4096x4096 x 4096x4096
- 16x8192x8192 x 8192x4096
- 64x4096x4096 x 4096x8192
- 8x8192x8192 x 8192x8192

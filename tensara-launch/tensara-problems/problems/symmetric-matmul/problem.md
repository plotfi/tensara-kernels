---
slug: "symmetric-matmul"
title: "Symmetric Matrix Multiplication"
difficulty: "MEDIUM"
author: "sarthak"
tags: ["matmul"]
---

Perform multiplication of two symmetric matrices:
$$
C[i][j] = \sum_{k=0}^{N-1} A[i][k] \cdot B[k][j]
$$

## Input
- $A$ is a symmetric matrix of size $N \times N$
- $B$ is a symmetric matrix of size $N \times N$ 

## Output
- Matrix $C = AB$ of size $N \times N$

## Notes:
- All matrices $\text{A}$, $\text{B}$, and $\text{C}$ are stored in row-major order
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/13_Matmul_for_symmetric_matrices.py)

## Test Case Sizes

- 4096x4096
- 6144x6144
- 7168x7168
- 8192x8192
- 9216x9216

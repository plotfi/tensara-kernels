---
slug: "matrix-vector"
title: "Matrix Vector Multiplication"
difficulty: "EASY"
author: "sarthak"
tags: ["matmul", "vector"]
---

Perform multiplication of a matrix and a vector:
$$
C[i] = \sum_{k=0}^{K-1} A[i][k] \cdot B[k]
$$

## Input:
- Matrix $A$ of size $M \times K$
- Vector $B$ of size $K \times 1$

## Output:
- Vector $C = AB$ of size $M \times 1$

## Notes:
- Matrix $\text{A}$ is stored in row-major order
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/4_Matrix_vector_multiplication_.py)

## Test Case Sizes

- M=4096, K=4096
- M=6144, K=4096
- M=7168, K=4096
- M=8192, K=4096
- M=9216, K=4096

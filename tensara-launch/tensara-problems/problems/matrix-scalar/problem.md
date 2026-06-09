---
slug: "matrix-scalar"
title: "Matrix Scalar Multiplication"
difficulty: "EASY"
author: "sarthak"
tags: ["matmul", "scalar"]
---

Perform multiplication of a matrix with a scalar value:
$$
C[i][j] = A[i][j] \cdot s
$$
where $s$ is the scalar value.

## Input:
- Matrix $A$ of size $\text{n} \times \text{n}$
- Scalar value $s$

## Output:
- Matrix $C = s \cdot A$ of size $\text{n} \times \text{n}$

## Notes:
- Matrix $\text{A}$ is stored in row-major order
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/5_Matrix_scalar_multiplication.py)

## Test Case Sizes

- 8192x8192 scalar=0.1
- 8192x8192 scalar=0.2
- 8192x8192 scalar=-0.3
- 8192x8192 scalar=0.4
- 8192x8192 scalar=-0.5
- 9216x9216 scalar=0.1
- 9216x9216 scalar=0.2
- 9216x9216 scalar=-0.3
- 9216x9216 scalar=0.4
- 9216x9216 scalar=-0.5

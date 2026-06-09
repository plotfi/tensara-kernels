---
slug: "tanh"
title: "Tanh"
difficulty: "EASY"
author: "sarthak"
tags: ["activation-function"]
---

Perform the Tanh activation function on an input matrix:
$$
C[i][j] = \text{tanh}(A[i][j])
$$

The Tanh function is defined as:
$$
\text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

## Input:
- Matrix $A$ of size $M \times N$ containing floating-point values

## Output:
- Matrix $C$ of size $M \times N$ containing the Tanh activation values

## Notes:
- Both matrices $\text{A}$ and $\text{C}$ are stored in row-major order
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/22_Tanh.py)

## Test Case Sizes

- 4096x4096
- 6144x4096
- 4096x7168
- 4096x8192
- 8192x8192

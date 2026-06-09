---
slug: "softmax"
title: "Softmax"
difficulty: "MEDIUM" 
author: "sarthak"
tags: ["activation-function", "normalization"]
---

Compute the softmax function over a specified dimension of an input tensor:
$$
\text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^{S_d} \exp(x_j)}
$$

where $x_i$ represents elements along the specified dimension $d$, and $S_d$ is the size of dimension $d$.

## Input:
- Tensor `input` of arbitrary shape $S_1 \times S_2 \times \cdots \times S_n$
- `dim` ($d$): Dimension to compute softmax over (0-based indexing)
- `shape`: Array containing the dimensions of the input tensor
- `ndim` ($n$): Number of dimensions in the input tensor

## Output:
- Tensor `output` with the same shape as input, containing the softmax probabilities

## Notes:
- The input tensor is stored in row-major order
- The output values should be in the range (0, 1)
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/23_Softmax.py)

## Test Case Sizes

- shape=(16, 128, 256), dim=1, dist=normal
- shape=(32, 512, 512), dim=2, dist=uniform
- shape=(8, 1024, 1024), dim=1, dist=normal
- shape=(64, 128, 128, 128), dim=2, dist=uniform
- shape=(4, 256, 256, 256), dim=3, dist=normal
- shape=(128, 10), dim=1, dist=normal
- shape=(256, 50, 50), dim=0, dist=uniform

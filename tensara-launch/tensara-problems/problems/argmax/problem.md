---
slug: "argmax"
title: "Argmax Over Dimension"
difficulty: "EASY" 
author: "sarthak"
tags: ["reduction"]
---

Find the indices of maximum values along a specified dimension of an input tensor:
$$
\text{output}[i_1,\ldots,i_{d-1},i_{d+1},\ldots,i_n] = \arg\max_{i_d} \text{input}[i_1,\ldots,i_d,\ldots,i_n]
$$

where $d$ is the dimension to perform argmax over, $n$ is the number of dimensions.

## Input:
- Tensor `input` of arbitrary shape $S_1 \times S_2 \times \cdots \times S_n$
- `dim` ($d$): Dimension to perform argmax over (0-based indexing)
- `shape`: Array containing the dimensions of the input tensor
- `ndim` ($n$): Number of dimensions in the input tensor

## Output:
- Tensor `output` with shape $S_1 \times \cdots \times S_{d-1} \times S_{d+1} \times \cdots \times S_n$
  - The output contains indices of maximum values along the specified dimension
  - The dimension being reduced is removed from the output shape

## Notes:
- The input tensor is stored in row-major order
- In case of ties (multiple maximum values), return the index of the first occurrence
- The output indices are 0-based
- The output tensor has one fewer dimension than the input tensor
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/51_Argmax_over_a_dimension.py)

## Test Case Sizes

- shape=(16, 128, 256), dim=1
- shape=(32, 512, 512), dim=0
- shape=(8, 1024, 1024), dim=2
- shape=(64, 128, 128, 128), dim=2
- shape=(4, 256, 256, 256), dim=1
- shape=(128, 64, 64, 64), dim=3

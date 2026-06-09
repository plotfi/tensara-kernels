---
slug: "group-norm"
title: "Group Normalization"
difficulty: "MEDIUM"
author: "sarthak"
tags: ["normalization"]
parameters:
  - name: "X"
    type: "float"
    pointer: "true"
    const: "true"
  
  - name: "Y"
    type: "float"
    pointer: "true"
    const: "false"

  - name: "B"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "N"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "G"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "spatial_size"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Implement Group Normalization for a tensor of shape `(batch_size, num_features, *)`.

Group Normalization divides the channels (features) into groups and computes normalization statistics within each group.

The formula for Group Normalization is:

$$
\text{y} = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
$$

where the mean $\mathrm{E}[x]$ and variance $\mathrm{Var}[x]$ are computed within each group of channels for each sample in the batch. $\gamma$ and $\beta$ are learnable parameters, and $\epsilon$ is a small value added to the variance for numerical stability.

## Input:
- Tensor $\text{X}$ of shape $(\text{B}, \text{N}, *)$ (input data)
- Number of groups to divide the channels into ($\text{G}$)

## Output:
- Tensor $\text{Y}$ with the same shape as input (normalized data)

## Notes:
- For this problem, take learnable parameters $\gamma = 1$ and $\beta = 0$.
- The number of channels (num_features) should be divisible by the number of groups.
- The mean and variance are computed across all spatial dimensions and the channels within each group.
- For each sample in the batch, the features are divided into $\text{N}$ groups, each with $\frac{\text{N}}{\text{G}}$ channels.
- Use $\epsilon = 10^{-5}$ for numerical stability.

## Test Case Sizes

- batch=16, features=64, groups=8, dims=[32, 32]
- batch=32, features=128, groups=16, dims=[64, 64]
- batch=8, features=32, groups=4, dims=[16, 16]
- batch=4, features=64, groups=8, dims=[32, 32, 32]
- batch=16, features=32, groups=8, dims=[]

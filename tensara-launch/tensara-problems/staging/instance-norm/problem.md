---
slug: "instance-norm"
title: "Instance Normalization"
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

  - name: "height"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "width"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Implement 2D Instance Normalization for a tensor of shape `(B, N, height, width)`.

Instance Normalization normalizes each channel/feature map of each sample in the batch independently. Unlike Batch Normalization which normalizes across the batch, Instance Normalization normalizes across spatial dimensions only, making it more suitable for tasks like style transfer where batch statistics should not matter.

The formula for Instance Normalization is:

$$
\text{y}_{n,c} = \frac{x_{n,c} - \mathrm{E}[x_{n,c}]}{\sqrt{\mathrm{Var}[x_{n,c}] + \epsilon}} * \gamma + \beta
$$

where the mean $\mathrm{E}[x_{n,c}]$ and variance $\mathrm{Var}[x_{n,c}]$ are computed over the spatial dimensions (height and width) for each feature map $c$ of each sample $n$ in the batch. $\epsilon$ is a small value added to the variance for numerical stability.

## Input:
- Tensor $\text{X}$ of shape $(\text{B}, \text{N}, \text{height}, \text{width})$ (input data)

## Output:
- Tensor $\text{Y}$ with the same shape as input (normalized data)

## Notes:
- For this problem, take learnable parameters $\gamma = 1$ and $\beta = 0$.
- For a 4D input tensor of shape $(\text{B}, \text{N}, \text{height}, \text{width})$, normalization is performed across height and width dimensions for each channel independently.
- Use $\epsilon = 10^{-5}$ for numerical stability.

## Test Case Sizes

- batch=16, features=64, height=32, width=32
- batch=32, features=128, height=64, width=64
- batch=8, features=32, height=16, width=16
- batch=4, features=16, height=128, width=128
- batch=64, features=3, height=224, width=224

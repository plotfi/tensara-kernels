---
slug: "max-pool-2d"
title: "2D Max Pooling"
difficulty: "MEDIUM" 
author: "sarthak"
tags: ["pooling"]
---

Perform 2D max pooling on an input tensor:
$$
\text{output}[i,j] = \max_{m=0,n=0}^{k-1,k-1} \text{input}[S \cdot i + D \cdot m - P, S \cdot j + D \cdot n - P]
$$

The max pooling operation slides a window of size $k \times k$ over the input tensor with stride $S$, dilation $D$, and padding $P$, computing the maximum value within each window position.

## Input:
- Matrix `input` of size $\text{H} \times \text{W}$ (input tensor)
- `kernel_size` ($k$): Size of the pooling window
- `stride` ($S$): Step size between window positions
- `padding` ($P$): Number of zero-padding elements added on all sides
- `dilation` ($D$): Spacing between kernel elements

## Output:
- Matrix `output` of size $\text{H}_{\text{out}} \times \text{W}_{\text{out}}$ where:
  $$\text{H}_{\text{out}} = \left\lfloor\frac{\text{H} + 2P - D(k-1) - 1}{S}\right\rfloor + 1$$
  $$\text{W}_{\text{out}} = \left\lfloor\frac{\text{W} + 2P - D(k-1) - 1}{S}\right\rfloor + 1$$

## Notes:
- All matrices are stored in row-major order
- Zero padding is applied when specified by the padding parameter
- For values outside the input boundaries (after padding), use negative infinity
- Dilation controls the spacing between kernel elements, creating an effective kernel size of $D(k-1) + 1$
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/42_Max_Pooling_2D.py)

## Test Case Sizes

- H=4096, W=4096, K=2, S=2, P=0, D=1
- H=8192, W=8192, K=3, S=2, P=1, D=1
- H=16384, W=16384, K=4, S=4, P=2, D=1
- H=1024, W=1024, K=3, S=3, P=1, D=2
- H=2048, W=2048, K=5, S=2, P=2, D=1
- H=4096, W=4096, K=7, S=3, P=3, D=1

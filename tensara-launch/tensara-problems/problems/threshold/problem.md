---
slug: "threshold"
title: "Image Thresholding"
difficulty: "EASY"
author: "sarthak"
tags: ["graphics"]
---

Perform binary thresholding on an input grayscale image:
$$
\text{Output}[i][j] = \begin{cases} 
255 & \text{if } \text{Input}[i][j] > \text{threshold\_value} \\
0 & \text{otherwise}
\end{cases}
$$

This operation converts a grayscale image into a binary (black and white) image by applying a threshold value.

## Input:
- Grayscale image of size $\text{height} \times \text{width}$
- Threshold value (typically in range [0, 255])

## Output:
- Binary image of size $\text{height} \times \text{width}$ with values 0 or 255

## Notes:
- The input tensor is a single-channel grayscale image
- In memory, the tensor is stored in row-major order
- The memory layout is [row0_col0, row0_col1, ..., row1_col0, ...]
- Pixel values are in the range [0, 255] for both input and output
- Output values are strictly binary: 0 for pixels below or equal to threshold, 255 for pixels above threshold

## Test Case Sizes

- 1024x768, threshold=64.0
- 1024x768, threshold=128.0
- 1024x768, threshold=192.0
- 1920x1080, threshold=64.0
- 1920x1080, threshold=128.0
- 1920x1080, threshold=192.0
- 3840x2160, threshold=64.0
- 3840x2160, threshold=128.0
- 3840x2160, threshold=192.0

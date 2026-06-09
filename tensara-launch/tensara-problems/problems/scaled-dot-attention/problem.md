---
slug: "scaled-dot-attention"
title: "Scaled Dot-Product Attention"
difficulty: "HARD" 
author: "sarthak"
tags: ["attention"]
---

Implement scaled dot-product attention, a key component of transformer architectures:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{E}}\right)V
$$

Where Q, K, and V are the query, key, and value matrices, and $E$ is the embedding dimension.

## Input:
- Matrix $Q$, $K$, $V$ of size $(\text{B} \times \text{H} \times \text{S} \times \text{E})$ (query, key, value)

## Output:
- Matrix $\text{output}$ of size $(\text{B} \times \text{H} \times \text{S} \times \text{E})$

## Notes:
- Input matrices are stored in a row-major format
- You can assume that the embedding dimension is the same for query, key and value matrices ($\text{E}$)
- Similarly, the number of heads is the same for query, key and value matrices ($\text{H}$)

## Test Case Sizes

- Batch=8, Heads=16, Seq_len=512, Embed_dim=64
- Batch=16, Heads=32, Seq_len=256, Embed_dim=64
- Batch=4, Heads=32, Seq_len=1024, Embed_dim=32
- Batch=32, Heads=8, Seq_len=512, Embed_dim=128
- Batch=8, Heads=16, Seq_len=512, Embed_dim=256
- Batch=8, Heads=16, Seq_len=2048, Embed_dim=64

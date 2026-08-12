# Kernel backlog — remaining CUDA kernels to implement

Ranked easiest→hardest by tensara difficulty (from `Problems.mhtml`: EASY/MEDIUM/
HARD), then clustered by similarity to each other and to already-implemented
kernels/frameworks. Reuse the existing frameworks — most clusters are "one
framework, many kernels."

**Existing frameworks to reuse**
- Dimension reductions: `kernel-implementation/dim-reduce.cuh`
- Global scalar reductions (losses): `block-reduce.cuh` + `loss.cuh` (`SmemTreeReduce`)
- Row-wise norms: `reduction.cuh`
- Activations / scalar ops: `activation.cuh`, `scalar-ops.cuh`
- Matmul + epilogue fusion: `gemm-epilogue.cuh`
- Matmul + prologue (dequant): `gemm-prologue.cuh`
- Pooling: `pooling.cuh`

> Note: `cosine-similarity`, `frobenius-norm`, `triplet-margin` are implemented in
> PR #38 (unmerged) — stubs on `main` but effectively done.

---

## EASY

### ✅ Dimension reductions — DONE (`dim-reduce.cuh`)
`sum-dim`, `mean-dim`, `max-dim`, `min-dim`, `product-dim`, `argmax`, `argmin`

### Global-reduction norms/stats — (frobenius/cosine in #38)
`frobenius-norm`, `cosine-similarity` — reuse `block-reduce.cuh` / `reduction.cuh`.

### Elementwise / image stencils — like `grayscale`, `threshold`, `conv-1d`
`matrix-scalar` (elementwise ×s), `box-blur` (2-D box stencil), `edge-detect`
(Sobel 2-D stencil), `histogram` (atomic scatter into bins)

### Simple / misc
`diagonal-matmul` (row-scale ≈ elementwise), `running-sum-1d` (1-D window ≈ `conv-1d`),
`ecc-point-negation` (batched modular negate ≈ elementwise)

### Low-precision dequant — prologue of `int8-weight-gemm` (`gemm-prologue.cuh`)
`mxfp4-dequantize`, `mxfp8-dequantize`

---

## MEDIUM

### Core matmul — `gemm-epilogue.cuh`, CS4803 lab2
`matrix-multiplication`, `square-matmul`, `symmetric-matmul`,
`upper-trig-matmul`, `lower-trig-matmul` (masked GEMM), `matrix-power` (repeated GEMM)

### Fused GEMM/conv + epilogue — directly `gemm-epilogue.cuh` + `scalar-ops.cuh`
`matmul-swish`, `matmul-sigmoid-sum`, `gemm-multiply-leakyrelu`, `conv2d-relu-hardswish`

### 2-D conv & pooling — dimensionality-up from 1-D (`pooling.cuh`, CS4803 lab3)
`conv-2d`, `avg-pool-2d`, `max-pool-2d`

### Per-row normalize with mean+var — extends `rms-norm`/`mean-subtract`
`layer-norm`, `batch-norm`

### Scan — new pattern (`running-sum-1d` is the warm-up)
`cumsum`, `cumprod`

### Low-precision quantize — inverse of dequant
`mxfp4-quantize`, `mxfp8-quantize`, `nvfp4-quantize`, `nvfp4-dequantize`

### Graphs — new pattern (iterative relaxation)
`all-pairs-shortest-path` (Floyd–Warshall), `shortest-path` (Bellman-Ford/Dijkstra),
`min-spanning-tree` (Prim)

### Finite-field / crypto — new pattern (needs modulus spec; currently test-uncovered)
`poly-multiply-ff`, `vector-multiply-ff`

### (`triplet-margin` — in #38, two-level reduce)

---

## HARD

### 3-D conv/pool — one more dimension than the 2-D cluster
`conv-square-3d`, `avg-pool-3d`, `max-pool-3d`

### Batched/high-dim matmul
`matmul-3d`, `matmul-4d`

### Low-precision GEMM — prologue-dequant matmul (`gemm-prologue.cuh`)
`mxfp4-gemm`, `mxfp8-gemm`, `nvfp4-gemm`, `nvfp4-gemv`

### Attention — matmul→softmax→matmul (uses `softmax` + GEMM; FlashAttention territory)
`scaled-dot-attention`

---

## Recommended order (max framework reuse first)
1. ✅ dim reductions → 2. norm/stats (merge #38; then layer-norm/batch-norm) →
3. fused GEMM epilogues → 4. core matmul → 5. 2-D conv/pool →
6. dequant → quantize → fp GEMM (one lineage) → 7. scan →
8. 3-D conv/pool + 3-D/4-D matmul → 9. graphs & finite-field → 10. attention.

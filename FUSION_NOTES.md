# Kernel Fusion Notes

Two kinds of kernel fusion are implemented in this repo. The distinction that
matters is **not** how many kernels you're merging — it's the *access pattern*
the consumer uses to read the producer's output.

| | Consumer reads... | Mechanism | Repo example |
|---|---|---|---|
| **Case 1 — map / pointwise fusion** | **one** producer output per element (1→1) | `__forceinline__` epilogue at the store | `gemm-epilogue.cuh` |
| **Case 2 — stencil / reduction fusion** | a **window / reduction** of producer outputs (many→1) | shared-memory staging + barrier (or recompute) | `conv1d-maxpool1d.cuh` |

Both are **producer→consumer** fusions (one dependency edge: A feeds B). "Two
producers" fusion means something else — see [Terminology](#terminology).

Case 1 has an **input-side mirror** worth naming separately: fuse a pointwise op
onto the compute's *inputs* at the load rather than its output at the store —
see [Prologue fusion](#prologue-fusion--the-input-side-mirror).

---

## Case 1 — map / pointwise (epilogue) fusion

**Kernel:** `kernel-implementation/gemm-epilogue.cuh`
**Uses:** `solutions-cuda/gemm-relu.cu`, `solutions-cuda/matmul-swish-scaling.cu`

A producer (GEMM) writes each output element, and a chain of *elementwise* ops
is applied to that element **before the store**:

```cpp
float acc = /* dot product (+ bias) */;
C[idx] = ep(acc);          // ep is a __forceinline__ functor: Relu, Compose<Swish, Scale>, ...
```

Because the consumer touches exactly one producer value per element, there is a
single scalar to transform — so `__forceinline__` inlines the whole epilogue
chain into the store. No extra kernel launch, the value stays in a register, and
the global round-trip an unfused "matmul then activation" would pay is gone.

**Why it's easy:** 1→1 access means no cross-thread/cross-element data is needed.
Composition (`Compose<F,G>`) is just function composition, all inlined.

**Verified (SASS, sm_89, `Compose<Swish,Scale>` variant):** 0 CALL instructions
(epilogue fully inlined), `__expf` → `MUFU.EX2`, a single `STG.E` store. (`Swish`
uses `__fdividef` to stay call-free; a plain `/` emits the IEEE-division helper.)

---

## Prologue fusion — the input-side mirror

Everything in Case 1 fuses a pointwise op onto the compute's *output* (the
epilogue). The mirror image is **prologue fusion**: fuse a pointwise op onto the
compute's *inputs*, applied as each operand is loaded, before it enters the math.

```
epilogue:  C[i,j] = ep( Σ_k A[i,k] · B[k,j] )          // op on the OUTPUT, at the store
prologue:  C[i,j] =    Σ_k pa(A[i,k]) · pb(B[k,j])     // op on the INPUTS, at the load
```

In `gemm-epilogue.cuh` a prologue would wrap the operand reads instead of the store:

```cpp
float av = pa(A[i * K + k]);                        // prologue on A
float bv = pb(B_T ? B[j * K + k] : B[k * N + j]);   // prologue on B
acc = fmaf(av, bv, acc);
```

**Canonical use case (why PyTorch Inductor added it): low-precision / weight-only
quantized matmuls.** The weight is stored quantized (int8/int4/fp8) and must be
dequantized before the matmul. Unfused, you materialize a full upcast fp16 weight
in DRAM, then matmul it. Prologue fusion folds the dequant into the load — you
read the small quantized tensor and expand it *in-kernel* at the point of use, so
the upcast tensor never exists in memory. That saves both the materialization and
the bandwidth (you read the 4–8× smaller tensor). TorchInductor's `mm`/`addmm`
Triton templates gained prologue fusion alongside the epilogue fusion they already
had, largely to make these dequant-matmuls fast; it is autotuned.

**Why prologue is trickier than epilogue** (the key asymmetry):

- An **epilogue** op runs **once per output element** — outputs are written once,
  so there is no redundancy. Pure win, always inlinable (that is why Case 1 is
  "easy").
- A **prologue** op runs on **inputs, which a tiled matmul re-reads many times**
  (each element of A feeds `N/BLOCK_N` output tiles, each element of B feeds
  `M/BLOCK_M`). Fusing it naively **recomputes** the op on every reload. So
  prologue fusion is really the **recompute-vs-stage tradeoff from Case 2**,
  applied to a pointwise op: recompute per tile-load, or apply it once when the
  tile is staged into shared memory and reuse it within the block.
- It usually still wins (dequant is cheap; bandwidth/materialization savings
  dominate; shared-memory staging bounds the recompute), but it raises
  register/shared-memory pressure — so frameworks **gate it behind
  heuristics/autotuning** rather than always applying it.

So in the taxonomy, a prologue is a **map fusion (1→1 pointwise) like the
epilogue**, but because it lands on *reused inputs* rather than *write-once
outputs*, it inherits Case 2's recompute/locality tradeoff — it sits between the
two.

**Worked example in this repo:** `kernel-implementation/gemm-prologue.cuh`
(kernel `int8-weight-gemm`) — a weight-only int8 **group**-quantized linear
layer, `C = A @ dequant(Wq, scale)ᵀ`. The dequant is a prologue functor
(`DequantInt8`) applied at the load, so the fp32 weight never exists in memory;
only the int8 `Wq` (4× smaller) and the group scales are read. Crucially the
group scale **varies along the reduction K**, so it *cannot* be hoisted out of
the sum into an epilogue — the dequant genuinely has to happen at the load,
which is what makes it a prologue. (The `mxfp4-gemm` / `nvfp4-gemm` kernels are
the same shape with block-scaled fp4/fp8 formats.)

It's the input-side analog of what `gemm-epilogue.cuh` does at the store: SASS
(sm_89) shows one kernel with `S8` loads of the int8 weight feeding straight
into `FFMA` — no intermediate fp32 weight buffer, no second kernel.

---

## Case 2 — stencil / window fusion

**Kernel:** `kernel-implementation/conv1d-maxpool1d.cuh`
**Uses:** `solutions-cuda/conv1d-maxpool1d.cu`

A same-length centered conv1d whose output is max-pooled:

```
producer: conv[c] = Σ_j in[c+j-rc] · w[j]        (rc = (K-1)/2, zero-padded)
consumer: out[o]  = max_{m<ks} conv[o·stride + m·dil - pad]
```

Now `out[o]` needs `ks` *different* `conv` values, and neighboring outputs share
most of them — there is no single scalar to inline. Two legal strategies:

- **Strategy A — stage in shared memory (what this kernel does).** Each block
  computes the conv outputs its pooled-output tile needs into shared memory
  (including the `±rc` **halo**), `__syncthreads()`, then reduces the pool
  windows out of shared memory. The N-length conv intermediate is **never**
  written to / read from global memory.
- **Strategy B — recompute.** Inline the producer at each consumer read; no
  shared memory, no barrier, but overlapping conv outputs get recomputed
  (~`ks/stride`×). Wins when the producer is cheap.

The hard part is the **footprint/halo bookkeeping**: a block producing pooled
outputs `[o0, o1)` needs conv outputs over
`[o0·stride − pad, (o1−1)·stride + (ks−1)·dil − pad]`, which need input over that
range `± rc`. Getting these ranges right (with stride/pad/dilation) is the whole
game — it's what frameworks automate.

**Verified (SASS, sm_89):** the conv intermediate uses `STS`/`LDS` (shared
store/load) with a single `BAR`; the **only** `STG` (global store) is the pooled
result. The intermediate never hits global memory — the point of the fusion.

**Reduction-fusion is the harder cousin:** if the consumer is a *reduction*
(e.g. softmax) rather than a fixed-width window, you also need a streaming/online
formulation (running max + sum) to fuse without materializing the intermediate —
that's what FlashAttention does.

---

## Terminology

This repo's fusions are **producer→consumer** (a linear chain `in → conv → pool →
out`). Watch the labels:

- **Map fusion** = 1→1 access (Case 1). Also called element-wise / injective fusion.
- **Epilogue fusion** = a map fused onto the compute's *output* (at the store).
  **Prologue fusion** = a map fused onto its *inputs* (at the load). Same pointwise
  op, opposite side; prologue carries a recompute cost because inputs are reused.
- **Stencil / reduction fusion** = many→1 access (Case 2). The consumer is a
  stencil (window) or a reduction.
- **Horizontal / sibling fusion** = two *independent* producers (no dependency)
  merged into one kernel, e.g. `sin(x)` and `cos(x)` together. This is the real
  "two producers" case — not what we did.
- **Diamond fusion** = two producers feeding a common consumer, `z = A(x)+B(x)`.

(An earlier draft mislabeled Case 2 "producer→producer"; it's producer→consumer
with two heavyweight stages.)

---

## Annotated references

Ordered by how directly each hits the pointwise-vs-window/reduction distinction.

### The taxonomy itself
- **TVM** — Chen et al., *OSDI 2018.* Classifies operators as **injective**
  (pointwise), **reduction**, **complex-out-fusable** (conv/matmul), **opaque**,
  and defines fusion rules between the categories. This is Case-1-vs-Case-2
  formalized. Best starting point.
- **DNNFusion** — Niu et al., *PLDI 2021.* Operator taxonomy by mapping type:
  **One-to-One** (pointwise), **One-to-Many**, **Many-to-Many** (stencil/
  reduction), etc.; fusion legality/profitability derived from these. The most
  explicit "1→1 vs many→1" treatment.
- **AStitch** — Zheng et al., *ASPLOS 2022.* Fusing memory-intensive ops across
  one-to-one / reduction / broadcast dependency patterns — when shared-memory
  stitching pays off.

### Stencil / window + the recompute-vs-stage tradeoff
- **Halide** — Ragan-Kelley et al., *PLDI 2013.* The canonical work; `compute_at`
  / `store_at` schedules *are* Strategy A vs B, with the halo/recomputation
  tradeoff. If you read one thing about Case 2, this.
- **Overlapped tiling for stencils** — Holewinski, Pouchet, Sadayappan, *ICS
  2012.* Formalizes the halo-recomputation Strategy B does by hand.
- **PPCG / polyhedral** — Verdoolaege et al., *ACM TACO 2013,* "Polyhedral
  parallel code generation for CUDA." The rigorous loop-fusion + tiling theory
  behind the footprint math.

### Reduction fusion (the hard subcase)
- **FlashAttention** — Dao et al., *NeurIPS 2022* (+ **FlashAttention-2**, 2023).
  Fusing matmul→softmax→matmul without materializing the S×S intermediate.
- **Online softmax** — Milakov & Gimelshein (NVIDIA, 2018). The running-max/sum
  trick that makes streaming reduction fusion possible. Read before FlashAttention.

### GPU tiling frameworks (automating the staging)
- **Triton** — Tillet, Kung, Cox, *MAPL 2019.* Tile abstraction that hides the
  shared-memory staging written by hand in `conv1d-maxpool1d.cuh`.
- **CUTLASS** (NVIDIA) — the reference for **epilogue fusion** (Case 1); see
  "epilogue visitor tree" for the composable-op pattern `Compose<>` mirrors.
  Its mixed-input GEMM examples are the prologue (input-side) analog.
- **TorchInductor** — Ansel et al., *"PyTorch 2,"* ASPLOS 2024. Inductor's `mm`/
  `addmm` templates do both **epilogue** and **prologue** fusion (the latter added
  mainly for dequant / weight-only-quantized matmuls); both are autotuned.

### Classic loop-fusion theory + surveys
- **Allen & Kennedy**, *Optimizing Compilers for Modern Architectures* — the
  loop-fusion/distribution chapter; fusion legality and fusion-preventing deps.
- **Kennedy & McKinley**, *1993* — classic loop fusion + distribution; note
  general loop fusion is NP-hard.
- **The Deep Learning Compiler: A Comprehensive Survey** — Li et al., *IEEE TPDS
  2021.* Where op fusion sits across XLA / TVM / Glow / etc.

**If you read three:** TVM + DNNFusion (the category-based fusion *rules*), and
Halide (the stencil recompute-vs-stage *mechanics* that `conv1d-maxpool1d`
implements by hand).

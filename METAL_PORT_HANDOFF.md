# Metal Port — Handoff for macOS validation

> Context dump for a fresh Claude Code session on macOS. The Metal backend of
> this repo was written entirely on a **Linux / NVIDIA** box where Metal cannot
> be compiled or run, so the `.metal` shaders and metal-cpp API calls are
> **unproven**. Your job: build the Metal side on macOS, fix what doesn't
> compile/link/run, and sanity-check correctness. The CUDA side is complete and
> fully tested — don't touch it (and you likely can't build it without CUDA).

## Where you are

- **Branch:** `metal-port` (branched off `refactor/solutions-and-tests`, which
  carries all the prior refactor work). 3 relevant commits on top:
  - `Make harness.cuh a cross-platform CUDA/Metal library`
  - `Add Metal shaders + solution() wrappers for all 86 kernels`
  - `Add build_metal.sh and document the cross-platform build`
- **Untracked** in the working tree: `micro-tensor/` (the known-good metal-cpp
  reference the design came from — keep it as a reference, it is NOT a
  dependency) and `resume.sh` (ignore).

## What this repo is

86 GPU "kernel problems". Each has:
- `tensara-harnesses/<k>.cu` — a **backend-agnostic** benchmark harness: seeds
  RNG, allocates `harness::Buffer<T>`s, fills inputs, `BENCHMARK(solution(...))`,
  previews output. No Metal code, no `#if`.
- `solutions/<k>.cu` — the **CUDA** `solution()` (a kernel launch).
- `solutions/<k>.metal` — the **Metal** shader; its kernel function is named
  `solution`.
- `solutions/<k>.cpp` — the **Metal** `solution()` wrapper (see below).

## How the cross-platform harness works (read this before editing)

`kernel-implementation/harness.cuh` picks a backend at compile time:
`__CUDACC__` → CUDA, `__APPLE__` → Metal, overridable with
`-DHARNESS_CUDA` / `-DHARNESS_METAL`. Backends live in
`kernel-implementation/detail/harness_{common,cuda,metal}.cuh`.

The key idea that keeps the harness identical on both backends:
- On CUDA, `harness::Buffer<T>` converts to a device `T*`.
- On Metal, `harness::Buffer<T>` wraps a **shared** `MTL::Buffer` and also
  converts to `T*` (the `contents()` pointer). A **registry**
  (`kernel-implementation/detail/harness_metal.cuh`) maps that pointer back to
  its `MTL::Buffer`.
- So the harness calls `solution(a, b, c, n)` the same way on both, and the
  `extern "C" void solution(const float*, ...)` declaration is identical.

On Metal, `solution()` is provided by `solutions/<k>.cpp`:
```cpp
#include "../kernel-implementation/harness.cuh"
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    auto pso = harness::pipeline("relu");                  // compiles relu.metal, fn "solution"
    uint32_t N = (uint32_t)(n * m);
    harness::dispatch(pso, { harness::buf(input), harness::buf(output) },
                      { harness::arg(N) }, n * m);         // bind buffers, then scalars; grid
}
```
- `harness::pipeline(name)` compiles `<shader_dir>/<name>.metal` (function
  `solution`), cached.
- `harness::buf(ptr)` recovers the `MTL::Buffer` for a Buffer's pointer.
- `harness::dispatch(pso, {buffers}, {scalars}, grid[, tpg])` binds buffers at
  indices `0..k-1`, scalars (`setBytes`) at `k..`, and `dispatchThreads` a 1-D
  grid.

The CUDA build links `<k>.cu`; the Metal build links `<k>.cpp` — never both — so
`solution()` is defined once per binary.

**Shared shader headers.** Metal's runtime source compiler has no include path
for local headers, so `harness_metal.cuh` inlines local `#include "x.metal.h"`
directives itself (from the shader dir) before compiling. The 10 activation
shaders use this: they all `#include "activation.metal.h"` (functor structs +
the templated `activation_x8` kernel body, a port of `activation.cuh`) and just
call `activation_x8<Functor>(...)`. `build_metal.sh` stages `*.metal.h` next to
the shaders. If an activation shader fails to compile, check the inlined source.

**metal-cpp implementation TU:** `NS_PRIVATE_IMPLEMENTATION` /
`MTL_PRIVATE_IMPLEMENTATION` are defined in `harness_metal.cuh`, **guarded by
`HARNESS_METAL_IMPL`**. `build_metal.sh` compiles the wrapper object with
`-DHARNESS_METAL_IMPL` and the harness object without it, so exactly one TU per
binary carries the metal-cpp implementation. If you see duplicate-symbol link
errors, this wiring is the thing to check.

## Build + run on macOS

```sh
# metal-cpp = the dir containing Metal/Metal.hpp and Foundation/Foundation.hpp
METAL_CPP=/path/to/metal-cpp ./build_metal.sh vector-addition   # one kernel
METAL_CPP=/path/to/metal-cpp ./build_metal.sh                   # all 86

HARNESS_SHADER_DIR=build/metal ./build/metal/vector-addition
```
Expected output shape:
```
=== vector-addition ===
Avg kernel time: 0.00xx ms (over 100 iters)
Output d_output (first 10): ...
Done.
```

## What is verified vs NOT

Verified on Linux:
- CUDA builds; `tests/run_tests.sh` → 22 pass, 0 regressions.
- All 86 `harness.cu + wrapper.cpp` pairs **compile + link against a *stub*
  metal-cpp** (validates the C++ structure: registry, dispatch, one `solution`,
  single-impl-TU wiring). This does NOT validate the real metal-cpp API or MSL.

NOT verified (your job):
- The `.metal` shaders compile under the Metal shader compiler.
- The real metal-cpp method names/signatures match the calls in
  `harness_metal.cuh` (I mirrored `micro-tensor/detail/metal_backend.h`, which
  ran on macOS — diff against it if something's off).
- Numerical correctness of any shader.
- That the single-impl-TU scheme links cleanly with real metal-cpp.

## Suggested plan

1. **Build `vector-addition` first.** Most failures here are in
   `harness_metal.cuh` (metal-cpp API) and apply to every kernel — fix once.
   Cross-check every metal-cpp call against `micro-tensor/detail/metal_backend.h`
   and `micro-tensor/examples/vector_add_metal.cpp` (known-good).
2. Then the other **real-MSL** kernels: `relu`, `grayscale`, `matrix-vector`,
   `conv-1d`, `avg-pool-1d`, then a reduction (`l2-norm`, `softmax`). The
   reductions dispatch **one threadgroup per row**: `grid = rows*256`, `tpg =
   256`, using `threadgroup_position_in_grid` as the row and a threadgroup tree
   reduce — verify those semantics on real Metal.
3. **Correctness:** `tests/` has CPU reference implementations (CUDA-only build).
   Easiest check on Metal is to eyeball `preview` output, or port a couple of
   `tests/test-<k>.cu` CPU references into a tiny Metal checker. The 22 real
   kernels and their exact formulas are in `solutions/<k>.metal` and mirrored in
   `tests/test-<k>.cu`.
4. The **64 stub** kernels build and dispatch a **no-op** shader (output stays
   zero) — that's expected; they mirror the CUDA `// TODO: implement` stubs.

## The 22 kernels with real MSL (worth validating)

`relu elu leaky-relu swish gelu selu sigmoid soft-plus tanh hard-sigmoid`
`vector-addition matrix-vector conv-1d grayscale avg-pool-1d`
`rms-norm l1-norm l2-norm max-normalize mean-subtract log-softmax softmax`

## Likely failure points (in priority order)

1. metal-cpp API signature mismatches in `harness_metal.cuh` (`newBuffer`,
   `newLibrary`, `newComputePipelineState`, `newFunction`, `setBuffer`,
   `setBytes`, `dispatchThreads`, `MTL::Size`, `NS::String::string`,
   `ResourceStorageModeShared`). Fix against micro-tensor.
2. `build_metal.sh` link flags — currently `-framework Metal -framework
   Foundation`. You may need `-framework QuartzCore` or `-fobjc-arc`; adjust.
3. MSL shader compile errors — math function namespaces, `half`, ternaries.
4. Duplicate metal-cpp symbols → the `HARNESS_METAL_IMPL` split (see above).
5. `dispatchThreads` requires non-uniform-threadgroup support; fine on Apple
   silicon. Reductions assume row length `D` is a multiple of 8-ish / that
   `tpg=256 >= D`; test sizes use `D=64`.

## Don't

- Don't add `micro-tensor` as a dependency (reference only).
- Don't edit the harnesses to add Metal `#if` branches — the whole point is that
  they stay backend-agnostic; all Metal logic lives in `solutions/<k>.cpp` +
  `harness_metal.cuh`.
- Don't change the CUDA `solutions/<k>.cu` or `tests/`.

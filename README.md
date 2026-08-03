# Kernel Benchmarks

CUDA kernel implementations with performance-measuring harnesses. Each harness runs a 3-iteration warmup followed by 100 timed iterations and reports the average kernel time.

## Repository Layout

```
kernel-harnesses/              # One backend-agnostic .cu harness per kernel problem
solutions-cuda/                # Per kernel: <k>.cu (CUDA solution)
solutions-metal/               # Per kernel: <k>.cpp + <k>.metal (Metal wrapper + shader)
kernel-implementation/         # Shared: harness.cuh (+ detail/ backends), reduction.cuh, activation.cuh
tests/                         # Correctness tests (CPU reference vs. GPU)
CMakeLists.txt                 # Build system (CUDA harnesses + tests via CTest; Metal on macOS)
build_metal.sh                 # Legacy standalone Metal build (macOS + metal-cpp)
run_all.sh                     # Run every built binary
```

Each harness includes `harness.cuh` and wraps its `solution()` call in the
`BENCHMARK(...)` macro, which runs the warmup + timed loop and prints the
average kernel time:

```c++
#include "../kernel-implementation/harness.cuh"
...
BENCHMARK(solution(d_a, d_b, d_c, m, n, k));
```

## Building

The build is CMake. Each harness/test executable lists **both** the harness and
its `solutions-cuda/<name>.cu` as sources, so editing a solution (or a shared
header) triggers a proper rebuild.

**Configure once, then build everything:**
```bash
cmake -S . -B build            # add -G Ninja if you have it
cmake --build build -j         # all harnesses + tests -> build/bin/<name>.exe
```
Implemented kernels link their in-repo code; every not-yet-implemented kernel
links a stub from `solutions-cuda/`, so it compiles and runs (producing zeroed
output) until you fill it in.

**Build a subset:**
```bash
cmake --build build -j -t harnesses                    # just the benchmark harnesses
cmake --build build -t matrix-multiplication           # one harness
```

By default `CMAKE_CUDA_ARCHITECTURES=native` (the building machine's GPU);
override with `cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89`.

### Filling in a kernel

**Every** kernel — implemented or not — lives at `solutions-cuda/<name>.cu` with the
correct `solution()` signature (kept in sync with the harness). Unimplemented
ones ship as a stub with an empty body. To implement (or change) a kernel, edit
its solution and rebuild:

```bash
$EDITOR solutions-cuda/matrix-multiplication.cu   # fill in / edit the body
cmake --build build -t matrix-multiplication
./build/bin/matrix-multiplication.exe
```

The harness and the test both link `solutions-cuda/<name>.cu` automatically — no
harness or test edits needed. Shared templates that several solutions build on
(`reduction.cuh` for the norm/softmax family, `activation.cuh` for the pointwise
activations) live in `kernel-implementation/`; a solution pulls one in with a
short `#include`, e.g. `solutions-cuda/rms-norm.cu` defines its macros and includes
`../kernel-implementation/reduction.cuh`.

## Running

**Run a single kernel:**
```bash
./build/bin/relu.exe
```

**Run all built kernels:**
```bash
./run_all.sh
```

Example output:
```
=== relu ===
Avg kernel time: 0.0015 ms (over 100 iters)
Output output (first 10): 0.000000 0.000000 0.381271 ...
Done.
```

## Cross-platform (CUDA / Metal)

`kernel-implementation/harness.cuh` is a cross-platform tensor/harness library
(design borrowed from `micro-tensor`, folded in so there's no external
dependency). It selects a backend at compile time — `__CUDACC__` → CUDA,
`__APPLE__` → Metal (via metal-cpp), overridable with `-DHARNESS_CUDA` /
`-DHARNESS_METAL` — and hides device allocation behind `harness::Buffer<T>`
(CUDA host+device pair, or a shared Metal `MTL::Buffer`). The backends live in
`kernel-implementation/detail/harness_{common,cuda,metal}.cuh`.

**The harness is backend-agnostic** — it just calls `solution(a, b, c, n)` and
contains no Metal code, no `#if`. On Metal, `harness::Buffer<T>` hands
`solution` the shared-memory pointer (like the CUDA device pointer), so the call
site is identical. Each kernel ships files across two directories:
- `solutions-cuda/<kernel>.cu` — CUDA `solution()` (a kernel launch).
- `solutions-metal/<kernel>.metal` — Metal shader whose kernel function is named `solution`.
- `solutions-metal/<kernel>.cpp` — Metal `solution()` wrapper: compiles the shader (`pipeline`),
  recovers each `MTL::Buffer` from its pointer (`harness::buf`), and dispatches.

The CUDA build links `<kernel>.cu`; the Metal build links `<kernel>.cpp` — never
both — so `solution()` is defined once per binary and the harness is unchanged.

**Build + run on Metal (macOS, needs metal-cpp headers):**
```bash
cmake -S . -B build -DMETAL_CPP=/path/to/metal-cpp
cmake --build build -t metal                              # all Metal kernels
TENSOR_SHADER_DIR=build/metal ./build/metal/relu          # shaders load from this dir
```
The standalone `./build_metal.sh` still works as a legacy fallback. (The Metal
CMake path is provided for parity but is verified on macOS only.)

The 22 implemented kernels (activations, vector-addition, matrix-vector,
conv-1d, the reductions, softmax, avg-pool-1d, grayscale) have real MSL shaders +
wrappers; the rest are stubs (`// TODO: implement`) to fill in, mirroring the
CUDA `solutions-cuda/` stubs. Metal is built/validated on macOS — this repo's CI
target is CUDA — but the Metal-side C++ (`harness_metal.cuh` + all 86
harness+wrapper pairs) is compile+link-checked.

## Testing

The `tests/` folder holds a correctness test per kernel. Each test builds a
small input, runs the kernel's `solution()`, computes a CPU reference, and
compares element-wise with a tolerance (loose for kernels using fast-math
intrinsics, tight for exact ones). A test prints `PASS`/`FAIL` and exits
nonzero on failure. `tests/test_utils.cuh` holds the shared device-buffer and
`compare`/`report` helpers.

Every test links `solutions-cuda/<kernel>.cu` — the same file the harness uses — so a
test runs your real code, whatever's in that file.

**Via CTest (preferred):** tests are registered with CMake. Tests whose solution
is still a stub are marked `DISABLED`, so a run is green by default and only
turns red on a real regression:

```bash
cmake --build build -j -t tests    # build every test
ctest --test-dir build -j          # run all enabled tests
ctest --test-dir build -R huber    # run one (regex on kernel name)
```

**Via the standalone runner** (also builds on demand, prints a full accounting):

```bash
./tests/run_tests.sh               # everything
./tests/run_tests.sh huber-loss    # one kernel, full output
```

It builds and runs one test per kernel — 74 in all — plus lists the 12 uncovered
kernels, reconciling to **every** kernel in `kernel-harnesses/` (currently 86):

```
=== tests (each links solutions-cuda/<kernel>.cu) ===
  PASS  avg-pool-1d
  FAIL  matrix-multiplication (stub)
  ...
=== uncovered (no test yet; needs exact spec) ===
  ----  mxfp4-gemm
  ...
Kernels total:     86
  passed:          22
  failed - stub:   52   (not implemented yet)
  failed - regr.:  0
  build errors:    0
  uncovered:       12
```

Each FAIL is tagged `(stub)` — the solution is still the empty stub, so implement
it — or `(regression)` — the solution is implemented but its output is wrong. The
**exit code is regressions + build errors**, so a run is "green" (exit 0) even
while stubs remain unimplemented; it only trips on something actually broken.

`SOLUTIONS_DIR` defaults to `../solutions`; point it elsewhere to test a
different set of solution files.

To test against a different set of solution files, `SOLUTIONS_DIR` still works
with the standalone runner:

```bash
SOLUTIONS_DIR=/path/to/your-solutions ./tests/run_tests.sh
```

To verify every test builds (signatures + reference code compile and link) in one
shot, just build the `tests` target:

```bash
cmake --build build -j -t tests
```

Some spec tests encode a **documented assumption** where the exact semantics
aren't pinned down by the repo (e.g. box-blur border handling, threshold as a
binary mask, histogram binning over `[0,1)`, loss reductions using the mean,
running-sum as a centered window, grayscale as Rec. 601 HWC). Each such file
states its assumption in a header comment — adjust the reference if your problem
differs. Graph tests (shortest-path, all-pairs, MST) construct non-negative,
well-defined inputs so the answer is unambiguous.

**Not yet covered** (need an exact numeric spec not present in the repo): the
micro-scaling FP formats `mxfp4-*`, `mxfp8-*`, `nvfp4-*`
(quantize/dequantize/gemm/gemv) and the finite-field `poly-multiply-ff` /
`vector-multiply-ff` (modulus unspecified). Provide the format/field spec and
these can be added.

To add a test: copy the closest `tests/test-<kernel>.cu`, declare
`extern "C" solution(...)`, and write the CPU reference. A plain
`tests/test-<kernel>.cu` is picked up automatically by CMake (and by
`tests/run_tests.sh` via its `TESTS` map); a test that shares a grouped source
with a `-D` flag needs an `add_kernel_test(...)` line in `CMakeLists.txt`. Each
test links `solutions-cuda/<kernel>.cu` automatically.

## Adding a New Kernel

Harnesses are built from the helpers in `kernel-implementation/harness.cuh`:

- `harness::begin("name")` — print the banner and seed the RNG.
- `harness::Buffer<T> buf(count)` — RAII host+device buffer (frees itself).
  - `buf.fill_random()` — fill inputs with random data (by type) and upload.
  - `buf.set({...})` — set explicit host values (e.g. a shape vector) and upload.
  - `buf.preview("label")` — copy back and print the first 10 elements.
  - implicitly converts to the device pointer, so pass it straight to `solution()`.
- `BENCHMARK(solution(...))` — warmup + timed loop + average time.

1. Create `kernel-harnesses/my-kernel.cu`. Use an existing harness as a
   template (e.g. `matrix-multiplication.cu`):
   ```c++
   #include "../kernel-implementation/harness.cuh"
   extern "C" void solution(const float* a, float* out, size_t n);

   int main() {
       harness::begin("my-kernel");
       size_t n = 1024;
       harness::Buffer<float> a(n), out(n);
       a.fill_random();
       BENCHMARK(solution(a, out, n));
       out.preview("out");
       printf("Done.\n");
       return 0;
   }
   ```

2. Implement `solution()` in `solutions-cuda/my-kernel.cu` (the harness and tests
   auto-link this by name).

3. Reconfigure so CMake globs the new harness, then build and run:
   ```bash
   cmake -S . -B build              # picks up kernel-harnesses/my-kernel.cu
   cmake --build build -t my-kernel
   ./build/bin/my-kernel.exe
   ```

## Generic Reduction Kernel

`kernel-implementation/reduction.cuh` is a one-block-per-row kernel that reduces
each row to a scalar, finalizes it, then rewrites each element as a function of
`(element, scalar)`. An instantiation defines five macros and includes it:

| Macro | Meaning | Default |
|---|---|---|
| `REDUCE_INIT` | reduction identity | `0.0f` |
| `COMBINE(A, B)` | reduction combine op | `(A) + (B)` |
| `MAP_OP(X)` | map element before reducing | required |
| `FINALIZE(ACC, D)` | finalize the reduced scalar (`D` = row length) | required |
| `WRITE_OP(X, ACC)` | per-element output transform | required |

Seven kernels are built on it (`solutions-cuda/{rms-norm,l1-norm,l2-norm,max-normalize,mean-subtract,log-softmax,softmax}.cu`):

| Kernel | `MAP_OP` | `COMBINE` | `FINALIZE` | `WRITE_OP` |
|---|---|---|---|---|
| rms-norm | `x*x` | sum | `rsqrt(acc/D + eps)` | `x * acc` |
| l1-norm | `\|x\|` | sum | `acc` | `x / acc` |
| l2-norm | `x*x` | sum | `sqrt(acc)` | `x / acc` |
| max-normalize | `\|x\|` | `fmaxf` | `acc` | `x / acc` |
| mean-subtract | `x` | sum | `acc / D` | `x - acc` |
| log-softmax | `exp(x)` | sum | `log(acc)` | `x - acc` |
| softmax | `exp(x)` | sum | `acc` | `exp(x) / acc` |

Assumes row length is a multiple of 4 (float4 vectorized) and `BLOCK_SIZE` is a
power of two. Build any of them directly, e.g. `cmake --build build -t l2-norm`.


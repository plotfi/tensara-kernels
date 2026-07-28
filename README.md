# Tensara Kernels

CUDA kernel implementations with performance-measuring harnesses. Each harness runs a 3-iteration warmup followed by 100 timed iterations and reports the average kernel time.

## Repository Layout

```
tensara-harnesses/             # One .cu harness per kernel problem
solutions/                     # One solution .cu per kernel (your code, or a fill-in stub)
kernel-implementation/         # Shared templates only (harness.cuh, reduction.cuh, activation.cuh)
tests/                         # Correctness tests (CPU reference vs. GPU)
Makefile                       # Build system
build_all.sh                   # Build every harness
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

**Build everything — all 86 harnesses build out of the box:**
```bash
./build_all.sh
```
Implemented kernels link their in-repo code; every not-yet-implemented kernel
links a stub from `solutions/`, so it compiles and runs (producing zeroed
output) until you fill it in.

**Build one harness:**
```bash
make build/bin/matrix-multiplication.exe        # auto-links solutions/matrix-multiplication.cu
make build/bin/matrix-multiplication.exe SOLUTION=my-solution.cu   # or point at your own file
```

### Filling in a kernel

**Every** kernel — implemented or not — lives at `solutions/<name>.cu` with the
correct `solution()` signature (kept in sync with the harness). Unimplemented
ones ship as a stub with an empty body. To implement (or change) a kernel, edit
its solution and rebuild:

```bash
$EDITOR solutions/matrix-multiplication.cu   # fill in / edit the body
make build/bin/matrix-multiplication.exe
./build/bin/matrix-multiplication.exe
```

The harness and the test both link `solutions/<name>.cu` automatically — no
harness or test edits needed. Shared templates that several solutions build on
(`reduction.cuh` for the norm/softmax family, `activation.cuh` for the pointwise
activations) live in `kernel-implementation/`; a solution pulls one in with a
short `#include`, e.g. `solutions/rms-norm.cu` defines its macros and includes
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

## Testing

The `tests/` folder holds a correctness test per kernel. Each test builds a
small input, runs the kernel's `solution()`, computes a CPU reference, and
compares element-wise with a tolerance (loose for kernels using fast-math
intrinsics, tight for exact ones). A test prints `PASS`/`FAIL` and exits
nonzero on failure. `tests/test_utils.cuh` holds the shared device-buffer and
`compare`/`report` helpers.

Every test links `solutions/<kernel>.cu` — the same file the harness uses — so a
test runs your real code, whatever's in that file. Just run:

```bash
./tests/run_tests.sh
```

It builds and runs one test per kernel — 74 in all — plus lists the 12 uncovered
kernels, reconciling to **every** kernel in `tensara-harnesses/` (currently 86):

```
=== tests (each links solutions/<kernel>.cu) ===
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

Build and run a single test against any solution file:

```bash
make -C tests build/bin/test-softmax.exe SOLUTION=/path/to/your-softmax.cu
./tests/build/bin/test-softmax.exe
```

To verify every spec test compiles (signatures + reference code) in one shot —
no linking, no GPU required:

```bash
make -C tests compile-check
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
`extern "C" solution(...)`, write the CPU reference, and add the kernel to
`SPEC_TESTS` in `tests/Makefile` and the `TESTS` map in `tests/run_tests.sh`. The
`tests/Makefile` pattern rule links `solutions/<kernel>.cu` automatically.

## Adding a New Kernel

Harnesses are built from the helpers in `kernel-implementation/harness.cuh`:

- `harness::begin("name")` — print the banner and seed the RNG.
- `harness::Buffer<T> buf(count)` — RAII host+device buffer (frees itself).
  - `buf.fill_random()` — fill inputs with random data (by type) and upload.
  - `buf.set({...})` — set explicit host values (e.g. a shape vector) and upload.
  - `buf.preview("label")` — copy back and print the first 10 elements.
  - implicitly converts to the device pointer, so pass it straight to `solution()`.
- `BENCHMARK(solution(...))` — warmup + timed loop + average time.

1. Create `tensara-harnesses/my-kernel.cu`. Use an existing harness as a
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

2. Implement `solution()` in a separate file (e.g. `my-solution.cu`).

3. Build and run:
   ```bash
   make build/bin/my-kernel.exe SOLUTION=my-solution.cu
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

Seven kernels are built on it (`solutions/{rms-norm,l1-norm,l2-norm,max-normalize,mean-subtract,log-softmax,softmax}.cu`):

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
power of two. Build any of them directly, e.g. `make build/bin/l2-norm.exe`.


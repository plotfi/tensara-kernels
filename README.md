# Tensara Kernels

CUDA kernel implementations with performance-measuring harnesses. Each harness runs a 3-iteration warmup followed by 100 timed iterations and reports the average kernel time.

## Repository Layout

```
tensara-harnesses/             # One .cu harness per kernel problem
kernel-implementation/         # Kernel implementations
kernel-implementation/harness.cuh  # Shared launch/benchmark helpers
Makefile                       # Build system
build_all.sh                   # Build every kernel
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

**Build any other harness against your own solution file:**
```bash
make build/bin/matrix-multiplication.exe SOLUTION=my-solution.cu
make build/bin/softmax.exe               SOLUTION=my-solution.cu
```

Note that you will need to include the solution header in a given harness, see everywhere I've included activations.cu ie:

```c++
#include "../kernel-implementation/activations.cu"
```

**Build everything (activation kernels built-in, others require SOLUTION):**
```bash
./build_all.sh
SOLUTION=my-solution.cu ./build_all.sh
```

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

Six kernels are built on it (`kernel-implementation/{rms-norm,l1-norm,l2-norm,max-normalize,mean-subtract,log-softmax}.cu`):

| Kernel | `MAP_OP` | `COMBINE` | `FINALIZE` | `WRITE_OP` |
|---|---|---|---|---|
| rms-norm | `x*x` | sum | `rsqrt(acc/D + eps)` | `x * acc` |
| l1-norm | `\|x\|` | sum | `acc` | `x / acc` |
| l2-norm | `x*x` | sum | `sqrt(acc)` | `x / acc` |
| max-normalize | `\|x\|` | `fmaxf` | `acc` | `x / acc` |
| mean-subtract | `x` | sum | `acc / D` | `x - acc` |
| log-softmax | `exp(x)` | sum | `log(acc)` | `x - acc` |

Assumes row length is a multiple of 4 (float4 vectorized) and `BLOCK_SIZE` is a
power of two. Build any of them directly, e.g. `make build/bin/l2-norm.exe`.


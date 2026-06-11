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

1. Create `tensara-harnesses/my-kernel.cu` with a `main()` that allocates data, calls `solution()` via `BENCHMARK(...)`, and prints results. Use an existing harness as a template (e.g. `matrix-multiplication.cu`). Include the shared helpers and declare but don't define `solution()`:
   ```c++
   #include "../kernel-implementation/harness.cuh"
   extern "C" void solution(/* your args */);
   ```

2. Implement `solution()` in a separate file (e.g. `my-solution.cu`).

3. Build and run:
   ```bash
   make build/bin/my-kernel.exe SOLUTION=my-solution.cu
   ./build/bin/my-kernel.exe
   ```


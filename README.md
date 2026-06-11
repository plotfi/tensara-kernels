# Tensara Kernels

CUDA kernel implementations with performance-measuring harnesses. Each harness runs a 3-iteration warmup followed by 100 timed iterations and reports the average kernel time.

## Repository Layout

```
tensara-harnesses/        # One .cu harness per kernel problem
kernel-implementation/    # Shared kernel implementations (activations.cu)
Makefile                  # Build system
build_all.sh              # Build every kernel
run_all.sh                # Run every built binary
```

## Building

**Build all activation kernels (self-contained, no solution file needed):**
```bash
make
```

**Build a specific activation kernel:**
```bash
make build/bin/relu.exe
make build/bin/gelu.exe
make build/bin/swish.exe
# etc.
```

**Build any other harness against your own solution file:**
```bash
make build/bin/matrix-multiplication.exe SOLUTION=my-solution.cu
make build/bin/softmax.exe               SOLUTION=my-solution.cu
```

**Build everything (activation kernels built-in, others require SOLUTION):**
```bash
./build_all.sh
SOLUTION=my-solution.cu ./build_all.sh
```

**Clean:**
```bash
make clean
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

### Generic kernel with your own solution

1. Create `tensara-harnesses/my-kernel.cu` with a `main()` that allocates data, calls `solution()`, and prints results. Use an existing harness as a template (e.g. `matrix-multiplication.cu`). Declare but don't define `solution()`:
   ```c
   extern "C" void solution(/* your args */);
   ```

2. Implement `solution()` in a separate file (e.g. `my-solution.cu`).

3. Build and run:
   ```bash
   make build/bin/my-kernel.exe SOLUTION=my-solution.cu
   ./build/bin/my-kernel.exe
   ```

### New activation using `activations.cu`

1. Add a `#define MY_ACT(X) ...` macro in `kernel-implementation/activations.cu`.

2. Add a `#elif defined(ACT_MY_ACT)` branch setting `ACTIVATION_NAME`, `BLOCK_SIZE`, `ACTIVATION`, and `SOLUTION()`.

3. Create `tensara-harnesses/my-act.cu` including `activations.cu` and declaring `solution()`:
   ```c
   #include "../kernel-implementation/activations.cu"
   extern "C" void solution(const float* input, float* output, size_t n, size_t m);
   ```

4. Add a per-target flag and entry to `ACTIVATIONS` in the `Makefile`:
   ```makefile
   ACTIVATIONS := ... my-act
   $(BINDIR)/my-act.exe: NVCCFLAGS += -DACT_MY_ACT
   ```

5. Add a line to `build_all.sh`:
   ```bash
   make build/bin/my-act.exe NVCCFLAGS='-O2 -std=c++17 -DACT_MY_ACT'
   ```

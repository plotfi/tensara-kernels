# CMake migration — handoff (for verifying the Metal build on macOS)

**PR:** #21 `build: convert to CMake (fixes solution-edit rebuilds); remove Makefiles`
(branch `cmake-build` on `github.com/plotfi/tensor-playground`).

**Why you're reading this on a Mac:** the CUDA path is written and fully verified
on Linux. The **Metal path is ported but UNTESTED** — there's no Metal toolchain
on the Linux box. This doc gives you everything to (a) understand the migration
and (b) verify/fix the `if(APPLE)` branch.

---

## 1. What motivated the change

The old `Makefile` pattern rule:

```make
$(BINDIR)/%.exe: $(HARNESS_DIR)/%.cu | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(if $(SOLUTION),$(SOLUTION),$(wildcard solutions-cuda/$*.cu))
```

The **solution** (`solutions-cuda/<k>.cu`) is pulled in via `$(wildcard)` *inside
the recipe*, so it was never a prerequisite. Editing a solution (this bit us on
`huber-loss.cu`) did **not** trigger a rebuild; neither did editing shared headers
(`tensor-lib/`, `kernel-implementation/`). You'd edit, rebuild, and run stale code.

CMake fixes this by listing **both** the harness and the solution as real sources
of each executable, so any edit (source or header, via nvcc's generated deps)
rebuilds correctly.

---

## 2. What the CMake build produces

Same layout the old scripts expect — binaries at `build/bin/<name>.exe`.

| Piece | How |
|---|---|
| **CUDA harnesses** | glob `kernel-harnesses/*.cu`; each executable = harness + `solutions-cuda/<name>.cu` |
| **Tests (CTest)** | 76 registered; test name = kernel name; stub-backed ones (`// TODO: implement`) marked `DISABLED` so a run is green by default |
| **Metal (macOS)** | `if(APPLE)` block, mirrors `build_metal.sh` — see §5 |

Grouped tests (shared source + `-D` flag) are registered explicitly in
`CMakeLists.txt`: activations (`test-activation.cu` + `-DACT_*`), `dim-reduce.cu`,
`argreduce.cu`, `loss-reduce.cu`, `trig-matmul.cu`, `avg-pool.cu`, `max-pool.cu`.
Everything else is picked up by a `test-*.cu` glob.

---

## 3. Files changed by the PR

- **Added:** `CMakeLists.txt`
- **Removed:** `Makefile`, `tests/Makefile`, `build_all.sh`
- **Updated:** `README.md` (build/test sections → cmake/ctest)
- **Kept:** `build_metal.sh` (standalone macOS fallback), `run_all.sh`,
  `tests/run_tests.sh` (still works; `./run_tests.sh <kernel>` runs one)

---

## 4. CUDA usage (verified on Linux, CUDA 12.4, sm_89)

```bash
cmake -S . -B build                 # add -G Ninja if available
cmake --build build -j              # all harnesses + tests -> build/bin/<name>.exe
cmake --build build -j -t harnesses # just harnesses
cmake --build build -t huber-loss   # one kernel
ctest --test-dir build -j          # correctness suite
ctest --test-dir build -R huber    # one test (regex)
```

Verified: touching `solutions-cuda/huber-loss.cu` recompiles+relinks; touching a
`tensor-lib` header rebuilds dependents; full ~164-target build OK; `ctest` =
**29/29 enabled passed, 47 stub-backed auto-skipped**.

Default `CMAKE_CUDA_ARCHITECTURES=native`; override with
`-DCMAKE_CUDA_ARCHITECTURES=89` (or `appleM`-irrelevant here).

---

## 5. Metal path — WHAT TO VERIFY ON THE MAC

This is the untested part. It's a translation of `build_metal.sh`. Build + run:

```bash
cmake -S . -B build -DMETAL_CPP=/path/to/metal-cpp
cmake --build build -t metal                          # all Metal kernels present
TENSOR_SHADER_DIR=build/metal ./build/metal/relu      # run one
```

### What the `if(APPLE)` block does (mirror of build_metal.sh)

For each kernel with both `solutions-metal/<k>.cpp` and `solutions-metal/<k>.metal`:

1. Executable `metal-<k>` from `kernel-harnesses/<k>.cu` + `solutions-metal/<k>.cpp`.
2. Both compiled **as C++** (`LANGUAGE CXX`, `-x c++`) with `-DTENSOR_METAL`.
3. Only the wrapper `.cpp` gets `-DTENSOR_METAL_IMPL` (it carries the metal-cpp
   implementation — the harness `.cu` must NOT, or you get duplicate symbols).
4. Include `-I<METAL_CPP>`; link `-framework Metal -framework Foundation`.
5. Output name is `<k>` (no `.exe`) into `build/metal/`.
6. `POST_BUILD` copies `solutions-metal/<k>.metal` next to the binary; shared
   `kernel-implementation/*.metal.h` are staged into `build/metal/` once.

### Reference: the shell build it mirrors

```bash
CXXFLAGS="-std=c++17 -O2 -x c++ -DTENSOR_METAL -I tensor-lib -I kernel-implementation -I$METAL_CPP"
clang++ $CXXFLAGS               -c kernel-harnesses/$k.cu   -o $OUT/$k.tensor.o
clang++ $CXXFLAGS -DTENSOR_METAL_IMPL -c solutions-metal/$k.cpp -o $OUT/$k.wrapper.o
clang++ $OUT/$k.tensor.o $OUT/$k.wrapper.o -o $OUT/$k -framework Metal -framework Foundation
cp solutions-metal/$k.metal $OUT/
```

### Specific things likely to need a check/fix

- **`-x c++` on a `.cu` file.** CMake may still try to treat `.cu` as CUDA
  despite `LANGUAGE CXX`. If you see nvcc being invoked or CUDA errors, the fix
  is to force the C++ compiler for these TUs (e.g. set the source language before
  `project()` sees CUDA, or symlink/copy the harness to a `.cpp`). This is the
  #1 risk.
- **Per-source `COMPILE_DEFINITIONS TENSOR_METAL_IMPL`** on the wrapper only —
  confirm the harness TU does *not* get it (duplicate metal-cpp symbols at link
  if it does).
- **Env var name:** runtime uses `TENSOR_SHADER_DIR` (the README's old
  `HARNESS_SHADER_DIR` was stale). Confirm the harness reads `TENSOR_SHADER_DIR`.
- **`METAL_CPP` default** is `${REPO}/metal-cpp` (the repo bundles it). Only pass
  `-DMETAL_CPP=` if yours is elsewhere.
- **Only ~22 kernels have real MSL** (activations, vector-addition, matrix-vector,
  conv-1d, reductions, softmax, avg-pool-1d, grayscale); the rest are stubs —
  expected.
- If the CMake Metal path fights you, `./build_metal.sh` still works standalone
  and is the ground truth to diff against.

### Definition of done on the Mac

- `cmake --build build -t metal` builds the ~22 implemented Metal kernels.
- `TENSOR_SHADER_DIR=build/metal ./build/metal/relu` runs and prints a sane
  benchmark line + output preview (compare against `./build_metal.sh relu`).
- If it needed fixes, push them onto the `cmake-build` branch (or a follow-up)
  and update this file's §5 with what was actually required.

---

## 6. One-screen cheatsheet

```bash
# configure
cmake -S . -B build [-G Ninja] [-DCMAKE_CUDA_ARCHITECTURES=89] [-DMETAL_CPP=/path]
# CUDA
cmake --build build -j                       # everything
cmake --build build -t <kernel>              # one harness
ctest --test-dir build -j                    # tests
# Metal (macOS)
cmake --build build -t metal
TENSOR_SHADER_DIR=build/metal ./build/metal/<kernel>
# still available
./tests/run_tests.sh [<kernel>]              # standalone runner
./build_metal.sh [<kernel>]                  # standalone Metal fallback
```

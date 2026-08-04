# Pooling Scalar-Op Generalization — Handoff

> Context dump for continuing in a fresh session (likely after a reboot to
> recover the GPU). This describes uncommitted work on a feature branch. The
> code compiles cleanly with nvcc but was **not** run because the GPU dropped
> off the bus mid-session.

## The guiding constraint (read first)

Generalize kernels in `kernel-implementation/` over the **scalar operation
only** — NOT over dimensionality. PR #8 (`refactor/generalize-kernels`, never
merged) went too far by adding `pool2d`/`pool3d`/`conv2d`/`conv3d`. The user
wants to write the 1D→2D→3D expansion themselves. So: template the inner scalar
op (avg vs max), keep the kernel at 1D.

## Where you are

- **Branch:** `generalize-pooling-scalar-op` (cut from `main` @ `379094c`).
- **State:** changes are **uncommitted** and **unvalidated on GPU**. Nothing
  pushed. Hold the commit until the CUDA tests pass.
- **Blocker:** `nvidia-smi` → "No devices were found" (RTX 4070 Ti dropped out;
  `/dev/nvidia*` nodes still present). It was healthy earlier this session and
  ran the full suite (22 pass). This is a recurring hardware flake cleared by a
  reboot — NOT a code problem. Symptoms of the dead device: the harness's
  unchecked `cudaMemcpy` leaves output all-zero (max-pool preview), and a
  managed-memory scratch test segfaults on host access.

## What was changed

Repo layout (post-refactor): backend-agnostic harnesses in `kernel-harnesses/`,
CUDA solutions in `solutions-cuda/`, Metal solutions in `solutions-metal/`
(`.cpp` wrapper + `.metal` shader), shared kernel headers in
`kernel-implementation/`, tensor library in `tensor-lib/` (namespace `tensor::`,
macros `TENSOR_CUDA`/`TENSOR_METAL`/`TENSOR_METAL_IMPL`/`TENSOR_SHADER_DIR`).

### New shared headers (1D only, op-templated)

**`kernel-implementation/pooling.cuh`**
```cpp
struct AvgPoolOp {
    static __device__ float init()                          { return 0.0f; }
    static __device__ float combine(float a, float b)       { return a + b; }
    static __device__ float finalize(float acc, int count)  { return acc / (float)count; }
};
struct MaxPoolOp {
    static __device__ float init()                          { return -CUDART_INF_F; }
    static __device__ float combine(float a, float b)       { return fmaxf(a, b); }
    static __device__ float finalize(float acc, int)        { return acc; }
};
template <typename Op>
__global__ void pool1d_kernel(const float* input, float* output,
                              int ks, int stride, int pad, int dilation,
                              int H, int Hout);   // one output elem per thread
inline int pool_out_size(int in, int ks, int pad, int dilation, int stride);
```
AvgPoolOp uses count_include_pad (divides by `ks`), matching the existing
avg-pool-1d semantics and the test reference.

**`kernel-implementation/pooling.metal.h`** — Metal mirror: same two functors
(with `-INFINITY`, `max`) + `template<typename Op> inline void pool1d(...)`.

### Removed (superseded by pooling.*)
- `kernel-implementation/avg-pool-1d.cuh`
- `kernel-implementation/avg-pool-1d.metal.h`

### Rewired / implemented solutions
- `solutions-cuda/avg-pool-1d.cu` → `pool1d_kernel<AvgPoolOp>` (was
  `avgpool1d_kernel`).
- `solutions-cuda/max-pool-1d.cu` → **implemented** `pool1d_kernel<MaxPoolOp>`
  (was a `// TODO: implement` stub). The stub had two bugs the op version fixes:
  `init` was `0.0f` (should be `-inf`); the Hout formula ignored `dilation`.
- `solutions-metal/avg-pool-1d.metal` → `#include "pooling.metal.h"` +
  `pool1d<AvgPoolOp>(in, out, ks, stride, pad, /*dilation=*/1, H, id)`.
- `solutions-metal/max-pool-1d.metal` → **implemented** `pool1d<MaxPoolOp>(...)`
  (buffers 0..1, scalars ks/stride/pad/dilation/H at 2..6).
- `solutions-metal/max-pool-1d.cpp` → **implemented** real dispatch (binds the
  two buffers + five scalars, grid = Hout).

### NOT touched (intentionally)
- Convolution (`conv-1d.cuh` etc.) — no scalar op to swap; leave 1D as-is.
- No pool2d/pool3d anywhere. That expansion is the user's to do.

## Signatures (for reference)

- `avg-pool-1d`: `solution(const float* in, int ks, int stride, int pad, float* out, size_t H)`
- `max-pool-1d`: `solution(const float* in, int ks, int stride, int pad, int dilation, float* out, size_t H)`
  (max-pool passes `dilation`; avg-pool does not — use dilation=1)

## Next steps (once the GPU is back)

```sh
# 1. Confirm the device is back
nvidia-smi -L                      # should list the RTX 4070 Ti

# 2. Run the two affected CUDA tests (both should PASS)
cd tests && ./run_tests.sh avg-pool-1d max-pool-1d
#   - avg-pool-1d must stay green (no regression from the rewire)
#   - max-pool-1d should now PASS (was TODO/skipped before)

# 3. Optional: full suite sanity (expect 23 pass now: was 22 + max-pool-1d)
cd tests && ./run_tests.sh

# 4. If green, commit on this branch and push
git add -A
git commit   # message: "Generalize 1-D pooling over the scalar op; implement max-pool-1d"
```

The test framework already registers both kernels
(`tests/max-pool.cu -DPOOL_DIM=1`, `tests/avg-pool.cu -DPOOL_DIM=1`); max-pool-1d
was previously skipped only because its solution contained `// TODO: implement`,
which the implementation removes.

## Metal note

The Metal side compiles structurally but, like the rest of the repo's Metal
backend, is unverifiable on Linux (see `METAL_PORT_HANDOFF.md`). `pooling.metal.h`
is staged into `build/metal/` automatically by the Makefile's `*.metal.h` copy
rule, so the runtime include-inliner resolves `#include "pooling.metal.h"`.

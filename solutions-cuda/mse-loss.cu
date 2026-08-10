// Solution stub for "mse-loss".
// The signature is derived from
// kernel-harnesses/mse-loss.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/mse-loss.exe
//   ./build/bin/mse-loss.exe
#include "../tensor-lib/tensor.cuh"
#include "../kernel-implementation/loss.cuh"

struct MseLossImpl {
  __device__ __forceinline__ static float apply(float x, float y) {
    float a = x - y;
    return a * a;
  }
};

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float *predictions, const float *targets,
                         float *output, const size_t *shape, size_t ndim) {
  // `shape` is a device pointer; the element count is the product of its dims.
  size_t scratch[8] = {0};
  size_t nd = ndim < 8 ? ndim : 8;
  const size_t* s = tensor::to_host(scratch, shape, nd);

  size_t n = 1;
  for (size_t i = 0; i < ndim; ++i)
    n *= s[i];

  launch_block_reduce<MseLossImpl, reduce_sum,
                      /*BLOCK_SIZE=*/512>(predictions, targets, output, n);
}

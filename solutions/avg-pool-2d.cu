// Solution stub for "avg-pool-2d".
// TODO: implement the body below. The signature is derived from
//       tensara-harnesses/avg-pool-2d.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/avg-pool-2d.exe
//   ./build/bin/avg-pool-2d.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H, size_t W) {
    // TODO: implement avg-pool-2d
}

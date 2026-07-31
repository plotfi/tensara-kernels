// Solution stub for "max-pool-3d".
// TODO: implement the body below. The signature is derived from
//       kernel-harnesses/max-pool-3d.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/max-pool-3d.exe
//   ./build/bin/max-pool-3d.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* input, int kernel_size, int stride, int padding, int dilation, float* output, size_t H, size_t W, size_t D) {
    // TODO: implement max-pool-3d
}

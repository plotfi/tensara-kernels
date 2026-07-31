// Solution stub for "batch-norm".
// TODO: implement the body below. The signature is derived from
//       kernel-harnesses/batch-norm.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/batch-norm.exe
//   ./build/bin/batch-norm.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* X, float* Y, size_t B, size_t F, size_t D1, size_t D2) {
    // TODO: implement batch-norm
}

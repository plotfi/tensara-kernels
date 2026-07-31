// Solution stub for "gemm-relu".
// TODO: implement the body below. The signature is derived from
//       kernel-harnesses/gemm-relu.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/gemm-relu.exe
//   ./build/bin/gemm-relu.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* A, const float* W, const float* b, float* C, size_t B, size_t N, size_t M) {
    // TODO: implement gemm-relu
}

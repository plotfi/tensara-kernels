// Solution stub for "gemm-multiply-leakyrelu".
// TODO: implement the body below. The signature is derived from
//       tensara-harnesses/gemm-multiply-leakyrelu.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/gemm-multiply-leakyrelu.exe
//   ./build/bin/gemm-multiply-leakyrelu.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* A, const float* B, const float* C, float alpha, float* output, size_t M, size_t N, size_t K) {
    // TODO: implement gemm-multiply-leakyrelu
}

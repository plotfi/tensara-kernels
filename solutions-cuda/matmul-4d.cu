// Solution stub for "matmul-4d".
// TODO: implement the body below. The signature is derived from
//       kernel-harnesses/matmul-4d.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/matmul-4d.exe
//   ./build/bin/matmul-4d.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* A, const float* B, float* C, size_t b, size_t i, size_t j, size_t l, size_t k) {
    // TODO: implement matmul-4d
}

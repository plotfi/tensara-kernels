// Solution stub for "matmul-3d".
// TODO: implement the body below. The signature is derived from
//       kernel-harnesses/matmul-3d.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/matmul-3d.exe
//   ./build/bin/matmul-3d.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* A, const float* B, float* C, size_t n, size_t m, size_t k, size_t l) {
    // TODO: implement matmul-3d
}

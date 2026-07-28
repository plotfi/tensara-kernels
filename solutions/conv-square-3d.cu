// Solution stub for "conv-square-3d".
// TODO: implement the body below. The signature is derived from
//       tensara-harnesses/conv-square-3d.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/conv-square-3d.exe
//   ./build/bin/conv-square-3d.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* A, const float* B, float* C, size_t size, size_t K) {
    // TODO: implement conv-square-3d
}

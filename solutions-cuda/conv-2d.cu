// Solution stub for "conv-2d".
// TODO: implement the body below. The signature is derived from
//       kernel-harnesses/conv-2d.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/conv-2d.exe
//   ./build/bin/conv-2d.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* A, const float* B, float* C, size_t H, size_t W, size_t Kh, size_t Kw) {
    // TODO: implement conv-2d
}

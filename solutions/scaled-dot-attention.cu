// Solution stub for "scaled-dot-attention".
// TODO: implement the body below. The signature is derived from
//       tensara-harnesses/scaled-dot-attention.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/scaled-dot-attention.exe
//   ./build/bin/scaled-dot-attention.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* Q, const float* K, const float* V, float* output, size_t B, size_t H, size_t S, size_t E) {
    // TODO: implement scaled-dot-attention
}

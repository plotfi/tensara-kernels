// Solution stub for "triplet-margin".
// TODO: implement the body below. The signature is derived from
//       kernel-harnesses/triplet-margin.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/triplet-margin.exe
//   ./build/bin/triplet-margin.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* anchor, const float* positive, const float* negative, float* loss, size_t B, size_t E, float margin) {
    // TODO: implement triplet-margin
}

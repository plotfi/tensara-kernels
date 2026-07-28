// Solution stub for "cosine-similarity".
// TODO: implement the body below. The signature is derived from
//       tensara-harnesses/cosine-similarity.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/cosine-similarity.exe
//   ./build/bin/cosine-similarity.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n, size_t d) {
    // TODO: implement cosine-similarity
}

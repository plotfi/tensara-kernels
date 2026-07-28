// Solution stub for "huber-loss".
// TODO: implement the body below. The signature is derived from
//       tensara-harnesses/huber-loss.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/huber-loss.exe
//   ./build/bin/huber-loss.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) {
    // TODO: implement huber-loss
}

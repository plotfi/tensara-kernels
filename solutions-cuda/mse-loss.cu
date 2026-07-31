// Solution stub for "mse-loss".
// TODO: implement the body below. The signature is derived from
//       kernel-harnesses/mse-loss.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/mse-loss.exe
//   ./build/bin/mse-loss.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* predictions, const float* targets, float* output, const size_t* shape, size_t ndim) {
    // TODO: implement mse-loss
}

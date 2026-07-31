// Solution stub for "threshold".
// TODO: implement the body below. The signature is derived from
//       kernel-harnesses/threshold.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/threshold.exe
//   ./build/bin/threshold.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* input_image, float threshold_value, float* output_image, size_t height, size_t width) {
    // TODO: implement threshold
}

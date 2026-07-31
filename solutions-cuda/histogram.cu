// Solution stub for "histogram".
// TODO: implement the body below. The signature is derived from
//       kernel-harnesses/histogram.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/histogram.exe
//   ./build/bin/histogram.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* image, int num_bins, float* histogram, size_t height, size_t width) {
    // TODO: implement histogram
}

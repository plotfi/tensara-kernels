// Solution stub for "edge-detect".
// TODO: implement the body below. The signature is derived from
//       tensara-harnesses/edge-detect.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/edge-detect.exe
//   ./build/bin/edge-detect.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* input_image, float* output_image, size_t height, size_t width) {
    // TODO: implement edge-detect
}

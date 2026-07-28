// Solution stub for "grayscale".
// TODO: implement the body below. The signature is derived from
//       tensara-harnesses/grayscale.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/grayscale.exe
//   ./build/bin/grayscale.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* rgb_image, float* grayscale_output, size_t height, size_t width, size_t channels) {
    // TODO: implement grayscale
}

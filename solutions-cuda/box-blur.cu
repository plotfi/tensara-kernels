// Solution stub for "box-blur".
// TODO: implement the body below. The signature is derived from
//       kernel-harnesses/box-blur.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/box-blur.exe
//   ./build/bin/box-blur.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* input_image, int kernel_size, float* output_image, size_t height, size_t width) {
    // TODO: implement box-blur
}

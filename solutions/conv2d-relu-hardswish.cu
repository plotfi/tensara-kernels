// Solution stub for "conv2d-relu-hardswish".
// TODO: implement the body below. The signature is derived from
//       tensara-harnesses/conv2d-relu-hardswish.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/conv2d-relu-hardswish.exe
//   ./build/bin/conv2d-relu-hardswish.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* image, const float* kernel, float* output, size_t H, size_t W, size_t Kh, size_t Kw) {
    // TODO: implement conv2d-relu-hardswish
}

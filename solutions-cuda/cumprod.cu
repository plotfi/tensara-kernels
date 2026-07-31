// Solution stub for "cumprod".
// TODO: implement the body below. The signature is derived from
//       kernel-harnesses/cumprod.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/cumprod.exe
//   ./build/bin/cumprod.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* input, float* output, size_t N) {
    // TODO: implement cumprod
}

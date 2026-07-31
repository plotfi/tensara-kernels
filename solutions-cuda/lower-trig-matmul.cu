// Solution stub for "lower-trig-matmul".
// TODO: implement the body below. The signature is derived from
//       kernel-harnesses/lower-trig-matmul.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/lower-trig-matmul.exe
//   ./build/bin/lower-trig-matmul.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t n) {
    // TODO: implement lower-trig-matmul
}

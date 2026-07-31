// Solution stub for "matrix-multiplication".
// TODO: implement the body below. The signature is derived from
//       kernel-harnesses/matrix-multiplication.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/matrix-multiplication.exe
//   ./build/bin/matrix-multiplication.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t n, size_t k) {
    // TODO: implement matrix-multiplication
}

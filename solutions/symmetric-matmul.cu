// Solution stub for "symmetric-matmul".
// TODO: implement the body below. The signature is derived from
//       tensara-harnesses/symmetric-matmul.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/symmetric-matmul.exe
//   ./build/bin/symmetric-matmul.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t n) {
    // TODO: implement symmetric-matmul
}

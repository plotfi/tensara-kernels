// Solution stub for "matmul-swish".
// TODO: implement the body below. The signature is derived from
//       tensara-harnesses/matmul-swish.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/matmul-swish.exe
//   ./build/bin/matmul-swish.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* input_matrix, const float* weight_matrix, const float* bias, float scaling_factor, float* output, size_t batch_size, size_t in_features, size_t out_features) {
    // TODO: implement matmul-swish
}

// Solution stub for "matrix-scalar".
// TODO: implement the body below. The signature is derived from
//       tensara-harnesses/matrix-scalar.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/matrix-scalar.exe
//   ./build/bin/matrix-scalar.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* input_matrix, float scalar, float* output_matrix, size_t n) {
    // TODO: implement matrix-scalar
}

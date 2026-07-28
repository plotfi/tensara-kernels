// Solution stub for "matrix-power".
// TODO: implement the body below. The signature is derived from
//       tensara-harnesses/matrix-power.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/matrix-power.exe
//   ./build/bin/matrix-power.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* input_matrix, size_t n, float* output_matrix, size_t size) {
    // TODO: implement matrix-power
}

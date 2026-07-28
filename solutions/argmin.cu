// Solution stub for "argmin".
// TODO: implement the body below. The signature is derived from
//       tensara-harnesses/argmin.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/argmin.exe
//   ./build/bin/argmin.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* input, int dim, int* output, const int* shape, int ndim) {
    // TODO: implement argmin
}

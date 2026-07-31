// Solution stub for "argmax".
// TODO: implement the body below. The signature is derived from
//       kernel-harnesses/argmax.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/argmax.exe
//   ./build/bin/argmax.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* input, int dim, int* output, const int* shape, int ndim) {
    // TODO: implement argmax
}

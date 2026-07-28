// Solution stub for "frobenius-norm".
// TODO: implement the body below. The signature is derived from
//       tensara-harnesses/frobenius-norm.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/frobenius-norm.exe
//   ./build/bin/frobenius-norm.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* X, float* Y, size_t size) {
    // TODO: implement frobenius-norm
}

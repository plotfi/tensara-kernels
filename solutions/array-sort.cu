// Solution stub for "array-sort".
// TODO: implement the body below. The signature is derived from
//       tensara-harnesses/array-sort.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/array-sort.exe
//   ./build/bin/array-sort.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const int* a, int* b, size_t n) {
    // TODO: implement array-sort
}

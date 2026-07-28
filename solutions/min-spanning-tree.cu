// Solution stub for "min-spanning-tree".
// TODO: implement the body below. The signature is derived from
//       tensara-harnesses/min-spanning-tree.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/min-spanning-tree.exe
//   ./build/bin/min-spanning-tree.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* A, float* min_weight, size_t n) {
    // TODO: implement min-spanning-tree
}

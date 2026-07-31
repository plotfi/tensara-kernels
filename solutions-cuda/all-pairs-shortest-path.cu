// Solution stub for "all-pairs-shortest-path".
// TODO: implement the body below. The signature is derived from
//       kernel-harnesses/all-pairs-shortest-path.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/all-pairs-shortest-path.exe
//   ./build/bin/all-pairs-shortest-path.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* adj_matrix, float* output, size_t n) {
    // TODO: implement all-pairs-shortest-path
}

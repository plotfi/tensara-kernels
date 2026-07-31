// Solution stub for "shortest-path".
// TODO: implement the body below. The signature is derived from
//       kernel-harnesses/shortest-path.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/shortest-path.exe
//   ./build/bin/shortest-path.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* d_adj_matrix, int source, float* d_distances, size_t n) {
    // TODO: implement shortest-path
}

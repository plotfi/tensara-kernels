// Solution stub for "running-sum-1d".
// TODO: implement the body below. The signature is derived from
//       kernel-harnesses/running-sum-1d.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/running-sum-1d.exe
//   ./build/bin/running-sum-1d.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* input, size_t W, float* output, size_t N) {
    // TODO: implement running-sum-1d
}

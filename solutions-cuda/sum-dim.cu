// Solution stub for "sum-dim".
// TODO: implement the body below. The signature is derived from
//       kernel-harnesses/sum-dim.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/sum-dim.exe
//   ./build/bin/sum-dim.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* input, int dim, float* output, const size_t* shape, size_t ndim) {
    // TODO: implement sum-dim
}

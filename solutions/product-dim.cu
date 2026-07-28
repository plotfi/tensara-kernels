// Solution stub for "product-dim".
// TODO: implement the body below. The signature is derived from
//       tensara-harnesses/product-dim.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/product-dim.exe
//   ./build/bin/product-dim.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* input, int dim, float* output, const size_t* shape, size_t ndim) {
    // TODO: implement product-dim
}

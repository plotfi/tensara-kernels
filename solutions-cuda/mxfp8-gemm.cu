// Solution stub for "mxfp8-gemm".
// TODO: implement the body below. The signature is derived from
//       kernel-harnesses/mxfp8-gemm.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/mxfp8-gemm.exe
//   ./build/bin/mxfp8-gemm.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const uint8_t* q_a, const uint8_t* scale_a, const uint8_t* q_b, const uint8_t* scale_b, float* c, size_t m, size_t n, size_t k) {
    // TODO: implement mxfp8-gemm
}

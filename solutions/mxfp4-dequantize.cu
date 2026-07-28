// Solution stub for "mxfp4-dequantize".
// TODO: implement the body below. The signature is derived from
//       tensara-harnesses/mxfp4-dequantize.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/mxfp4-dequantize.exe
//   ./build/bin/mxfp4-dequantize.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const uint8_t* q, const uint8_t* scale, float* out, size_t m, size_t k) {
    // TODO: implement mxfp4-dequantize
}

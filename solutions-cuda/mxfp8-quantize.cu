// Solution stub for "mxfp8-quantize".
// TODO: implement the body below. The signature is derived from
//       kernel-harnesses/mxfp8-quantize.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/mxfp8-quantize.exe
//   ./build/bin/mxfp8-quantize.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* a, uint8_t* q, uint8_t* scale, size_t m, size_t k) {
    // TODO: implement mxfp8-quantize
}

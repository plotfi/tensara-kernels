// Solution stub for "nvfp4-dequantize".
// TODO: implement the body below. The signature is derived from
//       kernel-harnesses/nvfp4-dequantize.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/nvfp4-dequantize.exe
//   ./build/bin/nvfp4-dequantize.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const uint8_t* q, const __nv_fp8_e4m3* scale, float sf_g, float* out, size_t m, size_t k) {
    // TODO: implement nvfp4-dequantize
}

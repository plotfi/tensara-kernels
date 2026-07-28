// Solution stub for "nvfp4-quantize".
// TODO: implement the body below. The signature is derived from
//       tensara-harnesses/nvfp4-quantize.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/nvfp4-quantize.exe
//   ./build/bin/nvfp4-quantize.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const half* a, float sf_g, uint8_t* q, __nv_fp8_e4m3* scale, size_t m, size_t k) {
    // TODO: implement nvfp4-quantize
}

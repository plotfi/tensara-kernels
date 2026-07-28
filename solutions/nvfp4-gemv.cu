// Solution stub for "nvfp4-gemv".
// TODO: implement the body below. The signature is derived from
//       tensara-harnesses/nvfp4-gemv.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/nvfp4-gemv.exe
//   ./build/bin/nvfp4-gemv.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const uint8_t* q_a, const __nv_fp8_e4m3* scale_a, float sf_g_a, const uint8_t* q_x, const __nv_fp8_e4m3* scale_x, float sf_g_x, half* y, size_t m, size_t k) {
    // TODO: implement nvfp4-gemv
}

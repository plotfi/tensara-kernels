// Metal solution wrapper stub for "nvfp4-gemm" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (harness::buf(ptr)) and scalars (harness::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const uint8_t* q_a, const __nv_fp8_e4m3* scale_a, float sf_g_a, const uint8_t* q_b, const __nv_fp8_e4m3* scale_b, float sf_g_b, half* c, size_t m, size_t n, size_t k) {
    auto pso = harness::pipeline("nvfp4-gemm");
    harness::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

// Metal solution wrapper stub for "mxfp4-gemm" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (harness::buf(ptr)) and scalars (harness::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const uint8_t* q_a, const uint8_t* scale_a, const uint8_t* q_b, const uint8_t* scale_b, float* c, size_t m, size_t n, size_t k) {
    auto pso = harness::pipeline("mxfp4-gemm");
    harness::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

// Metal solution wrapper stub for "mxfp8-quantize" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (harness::buf(ptr)) and scalars (harness::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* a, uint8_t* q, uint8_t* scale, size_t m, size_t k) {
    auto pso = harness::pipeline("mxfp8-quantize");
    harness::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

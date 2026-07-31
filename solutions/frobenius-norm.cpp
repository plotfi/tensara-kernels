// Metal solution wrapper stub for "frobenius-norm" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (harness::buf(ptr)) and scalars (harness::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* X, float* Y, size_t size) {
    auto pso = harness::pipeline("frobenius-norm");
    harness::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

// Metal solution wrapper stub for "scaled-dot-attention" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (harness::buf(ptr)) and scalars (harness::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* Q, const float* K, const float* V, float* output, size_t B, size_t H, size_t S, size_t E) {
    auto pso = harness::pipeline("scaled-dot-attention");
    harness::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

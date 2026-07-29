// Metal solution wrapper stub for "triplet-margin" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (harness::buf(ptr)) and scalars (harness::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* anchor, const float* positive, const float* negative, float* loss, size_t B, size_t E, float margin) {
    auto pso = harness::pipeline("triplet-margin");
    harness::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

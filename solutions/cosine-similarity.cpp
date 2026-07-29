// Metal solution wrapper stub for "cosine-similarity" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (harness::buf(ptr)) and scalars (harness::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n, size_t d) {
    auto pso = harness::pipeline("cosine-similarity");
    harness::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

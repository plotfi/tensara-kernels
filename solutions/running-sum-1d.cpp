// Metal solution wrapper stub for "running-sum-1d" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (harness::buf(ptr)) and scalars (harness::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input, size_t W, float* output, size_t N) {
    auto pso = harness::pipeline("running-sum-1d");
    harness::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

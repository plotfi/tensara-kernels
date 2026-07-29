// Metal solution wrapper stub for "cumsum" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (harness::buf(ptr)) and scalars (harness::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input, float* output, size_t N) {
    auto pso = harness::pipeline("cumsum");
    harness::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

// Metal solution wrapper stub for "min-spanning-tree" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (harness::buf(ptr)) and scalars (harness::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* A, float* min_weight, size_t n) {
    auto pso = harness::pipeline("min-spanning-tree");
    harness::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

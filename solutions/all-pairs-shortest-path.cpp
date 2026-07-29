// Metal solution wrapper stub for "all-pairs-shortest-path" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (harness::buf(ptr)) and scalars (harness::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* adj_matrix, float* output, size_t n) {
    auto pso = harness::pipeline("all-pairs-shortest-path");
    harness::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

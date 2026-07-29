// Metal solution wrapper stub for "shortest-path" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (harness::buf(ptr)) and scalars (harness::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* d_adj_matrix, int source, float* d_distances, size_t n) {
    auto pso = harness::pipeline("shortest-path");
    harness::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

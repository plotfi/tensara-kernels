// Metal solution wrapper stub for "matrix-scalar" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (harness::buf(ptr)) and scalars (harness::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input_matrix, float scalar, float* output_matrix, size_t n) {
    auto pso = harness::pipeline("matrix-scalar");
    harness::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

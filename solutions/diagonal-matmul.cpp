// Metal solution wrapper stub for "diagonal-matmul" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (harness::buf(ptr)) and scalars (harness::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* diagonal_a, const float* input_b, float* output_c, size_t n, size_t m) {
    auto pso = harness::pipeline("diagonal-matmul");
    harness::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

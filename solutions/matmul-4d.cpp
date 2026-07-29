// Metal solution wrapper stub for "matmul-4d" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (harness::buf(ptr)) and scalars (harness::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* A, const float* B, float* C, size_t b, size_t i, size_t j, size_t l, size_t k) {
    auto pso = harness::pipeline("matmul-4d");
    harness::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

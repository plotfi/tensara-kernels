// Metal solution wrapper stub for "matmul-sigmoid-sum" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (harness::buf(ptr)) and scalars (harness::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* A, const float* B, float* output, size_t M, size_t N, size_t K) {
    auto pso = harness::pipeline("matmul-sigmoid-sum");
    harness::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

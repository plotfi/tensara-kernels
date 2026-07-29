// Metal solution wrapper stub for "conv-square-3d" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (harness::buf(ptr)) and scalars (harness::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* A, const float* B, float* C, size_t size, size_t K) {
    auto pso = harness::pipeline("conv-square-3d");
    harness::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

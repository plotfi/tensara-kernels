// Metal solution wrapper stub for "avg-pool-3d" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (harness::buf(ptr)) and scalars (harness::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H, size_t W, size_t D) {
    auto pso = harness::pipeline("avg-pool-3d");
    harness::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

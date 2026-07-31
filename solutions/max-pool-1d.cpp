// Metal solution wrapper stub for "max-pool-1d" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (harness::buf(ptr)) and scalars (harness::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input, int kernel_size, int stride, int padding, int dilation, float* output, size_t H) {
    auto pso = harness::pipeline("max-pool-1d");
    harness::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

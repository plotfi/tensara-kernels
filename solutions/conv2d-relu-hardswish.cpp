// Metal solution wrapper stub for "conv2d-relu-hardswish" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (harness::buf(ptr)) and scalars (harness::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* image, const float* kernel, float* output, size_t H, size_t W, size_t Kh, size_t Kw) {
    auto pso = harness::pipeline("conv2d-relu-hardswish");
    harness::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

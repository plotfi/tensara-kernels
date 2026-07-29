// Metal solution wrapper stub for "threshold" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (harness::buf(ptr)) and scalars (harness::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input_image, float threshold_value, float* output_image, size_t height, size_t width) {
    auto pso = harness::pipeline("threshold");
    harness::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

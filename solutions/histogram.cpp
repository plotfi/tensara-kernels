// Metal solution wrapper stub for "histogram" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* image, int num_bins, float* histogram, size_t height, size_t width) {
    auto pso = tensor::pipeline("histogram");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

// Metal solution wrapper stub for "edge-detect" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input_image, float* output_image, size_t height, size_t width) {
    auto pso = tensor::pipeline("edge-detect");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

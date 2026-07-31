// Metal solution wrapper stub for "box-blur" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input_image, int kernel_size, float* output_image, size_t height, size_t width) {
    auto pso = tensor::pipeline("box-blur");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

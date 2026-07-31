// Metal solution wrapper stub for "conv2d-relu-hardswish" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* image, const float* kernel, float* output, size_t H, size_t W, size_t Kh, size_t Kw) {
    auto pso = tensor::pipeline("conv2d-relu-hardswish");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

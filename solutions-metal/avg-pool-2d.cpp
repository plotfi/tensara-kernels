// Metal solution wrapper stub for "avg-pool-2d" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H, size_t W) {
    auto pso = tensor::pipeline("avg-pool-2d");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

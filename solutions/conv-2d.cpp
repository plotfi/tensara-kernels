// Metal solution wrapper stub for "conv-2d" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* A, const float* B, float* C, size_t H, size_t W, size_t Kh, size_t Kw) {
    auto pso = tensor::pipeline("conv-2d");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

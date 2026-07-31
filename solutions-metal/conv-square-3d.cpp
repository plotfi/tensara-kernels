// Metal solution wrapper stub for "conv-square-3d" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* A, const float* B, float* C, size_t size, size_t K) {
    auto pso = tensor::pipeline("conv-square-3d");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

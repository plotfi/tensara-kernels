// Metal solution wrapper stub for "batch-norm" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* X, float* Y, size_t B, size_t F, size_t D1, size_t D2) {
    auto pso = tensor::pipeline("batch-norm");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

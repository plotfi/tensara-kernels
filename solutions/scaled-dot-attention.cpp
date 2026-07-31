// Metal solution wrapper stub for "scaled-dot-attention" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* Q, const float* K, const float* V, float* output, size_t B, size_t H, size_t S, size_t E) {
    auto pso = tensor::pipeline("scaled-dot-attention");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

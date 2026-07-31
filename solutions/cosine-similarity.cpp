// Metal solution wrapper stub for "cosine-similarity" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n, size_t d) {
    auto pso = tensor::pipeline("cosine-similarity");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

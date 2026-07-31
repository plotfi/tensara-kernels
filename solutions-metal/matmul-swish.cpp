// Metal solution wrapper stub for "matmul-swish" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input_matrix, const float* weight_matrix, const float* bias, float scaling_factor, float* output, size_t batch_size, size_t in_features, size_t out_features) {
    auto pso = tensor::pipeline("matmul-swish");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

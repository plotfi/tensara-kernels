// Metal solution wrapper stub for "matrix-scalar" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input_matrix, float scalar, float* output_matrix, size_t n) {
    auto pso = tensor::pipeline("matrix-scalar");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

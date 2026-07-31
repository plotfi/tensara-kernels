// Metal solution wrapper stub for "matrix-power" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input_matrix, size_t n, float* output_matrix, size_t size) {
    auto pso = tensor::pipeline("matrix-power");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

// Metal solution wrapper stub for "symmetric-matmul" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t n) {
    auto pso = tensor::pipeline("symmetric-matmul");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

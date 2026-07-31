// Metal solution wrapper stub for "matmul-3d" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* A, const float* B, float* C, size_t n, size_t m, size_t k, size_t l) {
    auto pso = tensor::pipeline("matmul-3d");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

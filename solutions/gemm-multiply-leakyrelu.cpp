// Metal solution wrapper stub for "gemm-multiply-leakyrelu" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* A, const float* B, const float* C, float alpha, float* output, size_t M, size_t N, size_t K) {
    auto pso = tensor::pipeline("gemm-multiply-leakyrelu");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

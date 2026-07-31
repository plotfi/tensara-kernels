// Metal solution wrapper stub for "gemm-relu" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* A, const float* W, const float* b, float* C, size_t B, size_t N, size_t M) {
    auto pso = tensor::pipeline("gemm-relu");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

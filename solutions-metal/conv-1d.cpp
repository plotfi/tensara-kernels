// Metal solution wrapper for "conv-1d".
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    auto pso = tensor::pipeline("conv-1d");
    uint32_t Nn = static_cast<uint32_t>(N), Kk = static_cast<uint32_t>(K);
    tensor::dispatch(pso, { tensor::buf(A), tensor::buf(B), tensor::buf(C) },
                      { tensor::arg(Nn), tensor::arg(Kk) }, N);
}

// Metal solution wrapper for "matrix-vector".
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t k) {
    auto pso = tensor::pipeline("matrix-vector");
    uint32_t M = static_cast<uint32_t>(m), K = static_cast<uint32_t>(k);
    tensor::dispatch(pso, { tensor::buf(input_a), tensor::buf(input_b), tensor::buf(output_c) },
                      { tensor::arg(M), tensor::arg(K) }, m);
}

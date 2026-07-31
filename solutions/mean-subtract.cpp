// Metal solution wrapper for "mean-subtract".
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* X, float* Y, size_t B, size_t D) {
    auto pso = tensor::pipeline("mean-subtract");
    uint32_t Dv = static_cast<uint32_t>(D);
    const size_t tpg = 256;
    tensor::dispatch(pso, { tensor::buf(X), tensor::buf(Y) }, { tensor::arg(Dv) }, B * tpg, tpg);
}

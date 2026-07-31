// Metal solution wrapper for "l2-norm".
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* X, float* Y, size_t B, size_t D) {
    auto pso = tensor::pipeline("l2-norm");
    uint32_t Dv = static_cast<uint32_t>(D);
    const size_t tpg = 256;
    tensor::dispatch(pso, { tensor::buf(X), tensor::buf(Y) }, { tensor::arg(Dv) }, B * tpg, tpg);
}

// Metal solution wrapper for "log-softmax".
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input, float* output, size_t M, size_t N) {
    auto pso = tensor::pipeline("log-softmax");
    uint32_t Dv = static_cast<uint32_t>(N);
    const size_t tpg = 256;
    tensor::dispatch(pso, { tensor::buf(input), tensor::buf(output) }, { tensor::arg(Dv) }, M * tpg, tpg);
}

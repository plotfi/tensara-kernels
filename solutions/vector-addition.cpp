// Metal solution wrapper for "vector-addition".
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* d_input1, const float* d_input2, float* d_output, size_t n) {
    auto pso = tensor::pipeline("vector-addition");
    uint32_t N = static_cast<uint32_t>(n);
    tensor::dispatch(pso, { tensor::buf(d_input1), tensor::buf(d_input2), tensor::buf(d_output) },
                      { tensor::arg(N) }, n);
}

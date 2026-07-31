// Metal solution wrapper for "leaky-relu" (unified activation kernel).
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input, float alpha, float* output, size_t n, size_t m) {
    auto pso = tensor::pipeline("leaky-relu");
    uint32_t N = static_cast<uint32_t>(n * m);
    size_t threads = (N + 7u) / 8u;
    tensor::dispatch(pso, { tensor::buf(input), tensor::buf(output) },
                      { tensor::arg(N), tensor::arg(alpha) }, threads);
}

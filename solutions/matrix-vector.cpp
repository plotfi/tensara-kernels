// Metal solution wrapper for "matrix-vector".
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t k) {
    auto pso = harness::pipeline("matrix-vector");
    uint32_t M = static_cast<uint32_t>(m), K = static_cast<uint32_t>(k);
    harness::dispatch(pso, { harness::buf(input_a), harness::buf(input_b), harness::buf(output_c) },
                      { harness::arg(M), harness::arg(K) }, m);
}

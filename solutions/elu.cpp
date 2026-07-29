// Metal solution wrapper for "elu".
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input, float* output, size_t n, size_t m, float alpha) {
    auto pso = harness::pipeline("elu");
    uint32_t N = static_cast<uint32_t>(n * m);
    harness::dispatch(pso, { harness::buf(input), harness::buf(output) },
                      { harness::arg(N), harness::arg(alpha) }, n * m);
}

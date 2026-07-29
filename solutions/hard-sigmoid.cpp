// Metal solution wrapper for "hard-sigmoid".
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    auto pso = harness::pipeline("hard-sigmoid");
    uint32_t N = static_cast<uint32_t>(n * m);
    harness::dispatch(pso, { harness::buf(input), harness::buf(output) }, { harness::arg(N) }, n * m);
}

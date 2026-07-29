// Metal solution wrapper for "log-softmax".
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input, float* output, size_t M, size_t N) {
    auto pso = harness::pipeline("log-softmax");
    uint32_t Dv = static_cast<uint32_t>(N);
    const size_t tpg = 256;
    harness::dispatch(pso, { harness::buf(input), harness::buf(output) }, { harness::arg(Dv) }, M * tpg, tpg);
}

// Metal solution wrapper for "rms-norm".
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* X, float* Y, size_t B, size_t D) {
    auto pso = harness::pipeline("rms-norm");
    uint32_t Dv = static_cast<uint32_t>(D);
    const size_t tpg = 256;
    harness::dispatch(pso, { harness::buf(X), harness::buf(Y) }, { harness::arg(Dv) }, B * tpg, tpg);
}

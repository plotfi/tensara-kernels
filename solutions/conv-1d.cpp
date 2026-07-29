// Metal solution wrapper for "conv-1d".
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    auto pso = harness::pipeline("conv-1d");
    uint32_t Nn = static_cast<uint32_t>(N), Kk = static_cast<uint32_t>(K);
    harness::dispatch(pso, { harness::buf(A), harness::buf(B), harness::buf(C) },
                      { harness::arg(Nn), harness::arg(Kk) }, N);
}

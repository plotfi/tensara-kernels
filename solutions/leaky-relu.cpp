// Metal solution wrapper for "leaky-relu" (unified activation kernel).
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input, float alpha, float* output, size_t n, size_t m) {
    auto pso = harness::pipeline("leaky-relu");
    uint32_t N = static_cast<uint32_t>(n * m);
    size_t threads = (N + 7u) / 8u;
    harness::dispatch(pso, { harness::buf(input), harness::buf(output) },
                      { harness::arg(N), harness::arg(alpha) }, threads);
}

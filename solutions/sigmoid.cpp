// Metal solution wrapper for "sigmoid" (unified activation kernel, 8 elems/thread).
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    auto pso = harness::pipeline("sigmoid");
    uint32_t N = static_cast<uint32_t>(n * m);
    float alpha = 0.0f;
    size_t threads = (N + 7u) / 8u;
    harness::dispatch(pso, { harness::buf(input), harness::buf(output) },
                      { harness::arg(N), harness::arg(alpha) }, threads);
}

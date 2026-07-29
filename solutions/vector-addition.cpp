// Metal solution wrapper for "vector-addition".
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* d_input1, const float* d_input2, float* d_output, size_t n) {
    auto pso = harness::pipeline("vector-addition");
    uint32_t N = static_cast<uint32_t>(n);
    harness::dispatch(pso, { harness::buf(d_input1), harness::buf(d_input2), harness::buf(d_output) },
                      { harness::arg(N) }, n);
}

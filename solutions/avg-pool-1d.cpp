// Metal solution wrapper for "avg-pool-1d".
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H) {
    auto pso = harness::pipeline("avg-pool-1d");
    int Hi = static_cast<int>(H);
    int Hout = (Hi + 2 * padding - kernel_size) / stride + 1;
    harness::dispatch(pso, { harness::buf(input), harness::buf(output) },
                      { harness::arg(kernel_size), harness::arg(stride), harness::arg(padding), harness::arg(Hi) },
                      static_cast<size_t>(Hout));
}

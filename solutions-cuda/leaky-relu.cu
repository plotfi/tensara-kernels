// Solution for "leaky-relu" (elementwise activation; takes an alpha parameter).
#include "../kernel-implementation/activation.cuh"

extern "C" void solution(const float* input, float alpha, float* output, size_t n, size_t m) {
    multi_solution<LeakyRelu, 512>(input, output, n, m, alpha);
}

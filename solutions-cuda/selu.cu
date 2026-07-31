// Solution for "selu" (elementwise activation).
#include "../kernel-implementation/activation.cuh"

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    multi_solution<selu, 1024>(input, output, n, m, 0.0f);
}

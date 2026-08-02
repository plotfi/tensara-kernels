// Solution for "elu" (elementwise activation; takes an alpha parameter).
#include "../kernel-implementation/activation.cuh"

extern "C" void solution(const float* input, float* output, size_t n, size_t m, float alpha) {
    multi_solution<Elu, 256>(input, output, n, m, alpha);
}

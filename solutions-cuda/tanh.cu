// Solution for "tanh" (elementwise activation).
#include "../kernel-implementation/activation.cuh"

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    multi_solution<tanh_act, 512>(input, output, n, m, 0.0f);
}

#include "../kernel-implementation/matrix-vector.cuh"

extern "C" void solution(const float* input_a, const float* input_b, float* output_c,
                         size_t m, size_t k) {
    int grid = static_cast<int>(m);
    int K = static_cast<int>(k);
    matvec_kernel<<<grid, MATVEC_BLOCK_SIZE>>>(input_a, input_b, output_c, K);
}

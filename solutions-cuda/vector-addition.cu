#include "../kernel-implementation/vector-addition.cuh"

extern "C" void solution(const float* d_input1, const float* d_input2,
                         float* d_output, size_t n) {
    int N = static_cast<int>(n);
    if (N == 0) return;

    const int BLOCK_SIZE = 256;
    int threads_needed = (N + 15) / 16;
    int grid = (threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;

    add_kernel_x16<<<grid, BLOCK_SIZE>>>(d_input1, d_input2, d_output, N);
}

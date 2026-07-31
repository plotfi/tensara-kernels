#include "../kernel-implementation/conv-1d.cuh"

extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    if (N == 0) return;

    int BLOCK_SIZE = 256;
    int grid = (static_cast<int>(N) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    conv1d_kernel<<<grid, BLOCK_SIZE>>>(A, B, C, N, K);
}

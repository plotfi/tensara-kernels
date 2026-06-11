#include <cuda_runtime.h>

__global__ void _kernel(const float* A, const float* B, float* C, size_t N, size_t K) {
    unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N) return;

    unsigned r = (K - 1) / 2;

    for (unsigned j = 0; j < K; ++j) {
        unsigned k = i + j - r;
        // if (k < 0 || k >= N)
        if (k >= N)
            continue;
        C[i] += A[k] * B[j];
    }
}

// Note: A, B, C are device pointers
extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    if (N == 0) return;

    int BLOCK_SIZE = 256;
    int grid = (static_cast<int>(N) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    _kernel<<<grid, BLOCK_SIZE>>>(A, B, C, N, K);
}

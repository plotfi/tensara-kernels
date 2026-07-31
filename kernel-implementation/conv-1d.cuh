#pragma once
#include <cuda_runtime.h>

__global__ void conv1d_kernel(const float* A, const float* B, float* C, size_t N, size_t K) {
    unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N) return;

    unsigned r = (K - 1) / 2;

    for (unsigned j = 0; j < K; ++j) {
        unsigned k = i + j - r;
        if (k >= N)
            continue;
        C[i] += A[k] * B[j];
    }
}

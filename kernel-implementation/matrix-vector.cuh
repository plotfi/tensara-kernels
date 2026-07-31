#pragma once
#include <cuda_runtime.h>

#define MATVEC_BLOCK_SIZE 128

__global__ void matvec_kernel(const float* A, const float* B, float* C, int K) {
    const float *row_a = A + blockIdx.x * K;

    float acc = 0.0f;

    const int K2 = K >> 1;

    const float2* row_a2 = reinterpret_cast<const float2*>(row_a);
    const float2* B2     = reinterpret_cast<const float2*>(B);

    for (int i = threadIdx.x; i < K2; i += MATVEC_BLOCK_SIZE) {
        float2 r = row_a2[i];
        float2 b = B2[i];
        acc = fmaf(r.x, b.x, acc);
        acc = fmaf(r.y, b.y, acc);
    }

    __shared__ float merge_buf[MATVEC_BLOCK_SIZE];

    for (int i = 1; i < MATVEC_BLOCK_SIZE; i *= 2) {
        merge_buf[threadIdx.x] = acc;
        __syncthreads();

        acc += merge_buf[(threadIdx.x + i) % MATVEC_BLOCK_SIZE];
        __syncthreads();
    }

    C[blockIdx.x] = acc;
}

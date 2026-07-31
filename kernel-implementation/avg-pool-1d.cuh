#pragma once
#include <cuda_runtime.h>

__global__ void avgpool1d_kernel(const float* input, float* output, int k, int S, int P, int H, int Hout, float inv_k) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= H) return;
    if (i >= Hout) return;

    float sum = 0.0f;
    for (int m = 0; m < k; m++) {
        int RI = S * i + m - P;
        if (0 <= RI && RI < H) {
            sum += input[RI];
        }
    }

    output[i] = inv_k * sum;
}

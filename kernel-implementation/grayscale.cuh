#pragma once
#include <cuda_runtime.h>

#define GRAYSCALE_BLOCK_SIZE 256

__global__ void grayscale_kernel(const float* rgb_image, float* grayscale_output, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    int out_base = i * 4;
    if (out_base >= N) return;

    if (out_base + 3 < N) {
        int f4_base = i * 3;

        float4 pixel1 = reinterpret_cast<const float4*>(rgb_image)[f4_base];
        float4 pixel2 = reinterpret_cast<const float4*>(rgb_image)[f4_base + 1];
        float4 pixel3 = reinterpret_cast<const float4*>(rgb_image)[f4_base + 2];

        float r0 = pixel1.x;
        float g0 = pixel1.y;
        float b0 = pixel1.z;

        float r1 = pixel1.w;
        float g1 = pixel2.x;
        float b1 = pixel2.y;

        float r2 = pixel2.z;
        float g2 = pixel2.w;
        float b2 = pixel3.x;

        float r3 = pixel3.y;
        float g3 = pixel3.z;
        float b3 = pixel3.w;

        float gray0 = 0.299f * r0 + 0.587f * g0 + 0.114f * b0;
        float gray1 = 0.299f * r1 + 0.587f * g1 + 0.114f * b1;
        float gray2 = 0.299f * r2 + 0.587f * g2 + 0.114f * b2;
        float gray3 = 0.299f * r3 + 0.587f * g3 + 0.114f * b3;

        float4 out = make_float4(gray0, gray1, gray2, gray3);
        *reinterpret_cast<float4*>(grayscale_output + out_base) = out;
    } else {
        for (int p = out_base; p < N; ++p) {
            int b = p * 3;
            grayscale_output[p] = 0.299f * rgb_image[b]
                                + 0.587f * rgb_image[b + 1]
                                + 0.114f * rgb_image[b + 2];
        }
    }
}

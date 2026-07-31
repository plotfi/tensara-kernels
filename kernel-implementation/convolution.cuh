#pragma once
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Centered cross-correlation with zero padding, 1-D through 3-D.
// Each kernel computes one output element per thread.
// ---------------------------------------------------------------------------

// --- 1-D:  C[i] = sum_j A[i+j-r] * B[j],  r = (K-1)/2 ---

__global__ void conv1d_kernel(const float* A, const float* B, float* C,
                              size_t N, size_t K) {
    unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N) return;

    unsigned r = (K - 1) / 2;
    float acc = 0.0f;
    for (unsigned j = 0; j < K; ++j) {
        unsigned k = i + j - r;
        if (k < N) acc += A[k] * B[j];
    }
    C[i] = acc;
}

// --- 2-D:  C[h,w] = sum_{jh,jw} A[h+jh-rh, w+jw-rw] * B[jh,jw] ---

__global__ void conv2d_kernel(const float* A, const float* B, float* C,
                              size_t H, size_t W, size_t Kh, size_t Kw) {
    unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned oh = idx / W, ow = idx % W;
    if (oh >= H) return;

    int rh = (static_cast<int>(Kh) - 1) / 2;
    int rw = (static_cast<int>(Kw) - 1) / 2;
    float acc = 0.0f;
    for (unsigned jh = 0; jh < Kh; ++jh)
        for (unsigned jw = 0; jw < Kw; ++jw) {
            int ih = static_cast<int>(oh) + static_cast<int>(jh) - rh;
            int iw = static_cast<int>(ow) + static_cast<int>(jw) - rw;
            if (ih >= 0 && ih < static_cast<int>(H) &&
                iw >= 0 && iw < static_cast<int>(W))
                acc += A[ih * W + iw] * B[jh * Kw + jw];
        }
    C[idx] = acc;
}

// --- 3-D (cube):  square kernel K×K×K on a size×size×size volume ---

__global__ void conv3d_kernel(const float* A, const float* B, float* C,
                              size_t S, size_t K) {
    unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned S2 = S * S;
    unsigned oz = idx / S2, rem = idx % S2;
    unsigned oy = rem / S, ox = rem % S;
    if (oz >= S) return;

    int r = (static_cast<int>(K) - 1) / 2;
    float acc = 0.0f;
    for (unsigned jz = 0; jz < K; ++jz)
        for (unsigned jy = 0; jy < K; ++jy)
            for (unsigned jx = 0; jx < K; ++jx) {
                int iz = static_cast<int>(oz) + static_cast<int>(jz) - r;
                int iy = static_cast<int>(oy) + static_cast<int>(jy) - r;
                int ix = static_cast<int>(ox) + static_cast<int>(jx) - r;
                if (iz >= 0 && iz < static_cast<int>(S) &&
                    iy >= 0 && iy < static_cast<int>(S) &&
                    ix >= 0 && ix < static_cast<int>(S))
                    acc += A[(iz * S + iy) * S + ix] * B[(jz * K + jy) * K + jx];
            }
    C[idx] = acc;
}

// frobenius-norm: Y = X / ||X||_F,  ||X||_F = sqrt(sum_i X[i]^2).
// A GLOBAL reduction (sum of squares) followed by a broadcast/normalize pass —
// the two-pass shape. Pass 1 reduces to a scalar with the shared SmemTreeReduce
// + one atomic per block; pass 2 rescales every element.
#include <cuda_runtime.h>
#include "../kernel-implementation/reduction.cuh"   // SmemTreeReduce, reduce_sum, id_zero

#define BLOCK 512

// Pass 1: scratch[0] = sum_i X[i]^2   (float4-vectorized + scalar tail).
__global__ void sumsq_kernel(const float* __restrict__ X, float* scratch, size_t n) {
    size_t base = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x) * 4;
    float r = 0.0f;
    if (base + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(X + base);
        r = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    } else {
        for (size_t i = base; i < n; ++i) r += X[i] * X[i];
    }
    r = SmemTreeReduce<BLOCK, reduce_sum, id_zero>::apply(r);   // all threads call uniformly
    if (threadIdx.x == 0) atomicAdd(scratch, r);
}

// Pass 2: Y[i] = X[i] * rsqrt(sum_sq) = X[i] / ||X||_F.
__global__ void normalize_kernel(const float* __restrict__ X, float* __restrict__ Y,
                                 const float* scratch, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) Y[i] = X[i] * rsqrtf(scratch[0]);
}

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* X, float* Y, size_t size) {
    static float* scratch = nullptr;              // one-float device accumulator (reused)
    if (!scratch) cudaMalloc(&scratch, sizeof(float));
    cudaMemset(scratch, 0, sizeof(float));

    int g1 = ((size + 3) / 4 + BLOCK - 1) / BLOCK;
    sumsq_kernel<<<g1, BLOCK>>>(X, scratch, size);
    int g2 = (size + BLOCK - 1) / BLOCK;
    normalize_kernel<<<g2, BLOCK>>>(X, Y, scratch, size);
}

#include <cuda_runtime.h>

// RMS-norm: y = x / sqrt(mean(x^2) + eps)
#define MAP_OP(X)         ((X) * (X))
#define FINALIZE(ACC, D)  rsqrtf(__fdividef(ACC, D) + 1e-5f)
#define WRITE_OP(X, ACC)  ((X) * (ACC))
#include "reduction.cuh"

// Note: X, Y are device pointers
extern "C" void solution(const float* X, float* Y, size_t B, size_t N) {
    launch_reduction(X, Y, static_cast<int>(B), static_cast<int>(N));
}

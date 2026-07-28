#include <cuda_runtime.h>

// L2-norm: y = x / sqrt(sum(x^2))
#define MAP_OP(X)         ((X) * (X))
#define FINALIZE(ACC, D)  sqrtf(ACC)
#define WRITE_OP(X, ACC)  __fdividef(X, ACC)
#include "../kernel-implementation/reduction.cuh"

// Note: X, Y are device pointers
extern "C" void solution(const float* X, float* Y, size_t B, size_t D) {
    launch_reduction(X, Y, static_cast<int>(B), static_cast<int>(D));
}

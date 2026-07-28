#include <cuda_runtime.h>

// L1-norm: y = x / sum(|x|)
#define MAP_OP(X)         fabsf(X)
#define FINALIZE(ACC, D)  (ACC)
#define WRITE_OP(X, ACC)  __fdividef(X, ACC)
#include "../kernel-implementation/reduction.cuh"

// Note: X, Y are device pointers
extern "C" void solution(const float* X, float* Y, size_t B, size_t D) {
    launch_reduction(X, Y, static_cast<int>(B), static_cast<int>(D));
}

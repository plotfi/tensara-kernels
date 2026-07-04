#include <cuda_runtime.h>

// Mean-subtract: y = x - mean(x)
#define MAP_OP(X)         (X)
#define FINALIZE(ACC, D)  __fdividef(ACC, D)
#define WRITE_OP(X, ACC)  ((X) - (ACC))
#include "reduction.cuh"

// Note: X, Y are device pointers
extern "C" void solution(const float* X, float* Y, size_t B, size_t D) {
    launch_reduction(X, Y, static_cast<int>(B), static_cast<int>(D));
}

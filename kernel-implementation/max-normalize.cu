#include <cuda_runtime.h>

// Max-normalize: y = x / max(|x|)   (infinity-norm normalization)
#define REDUCE_INIT       0.0f
#define COMBINE(A, B)     fmaxf(A, B)
#define MAP_OP(X)         fabsf(X)
#define FINALIZE(ACC, D)  (ACC)
#define WRITE_OP(X, ACC)  __fdividef(X, ACC)
#include "reduction.cuh"

// Note: X, Y are device pointers
extern "C" void solution(const float* X, float* Y, size_t B, size_t D) {
    launch_reduction(X, Y, static_cast<int>(B), static_cast<int>(D));
}

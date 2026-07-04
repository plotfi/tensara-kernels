#include <cuda_runtime.h>

// Log-softmax: y = x - log(sum(exp(x)))
#define MAP_OP(X)         __expf(X)
#define FINALIZE(ACC, D)  logf(ACC)
#define WRITE_OP(X, ACC)  ((X) - (ACC))
#include "reduction.cuh"

// Note: input, output are device pointers
extern "C" void solution(const float* input, float* output, size_t M, size_t N) {
    launch_reduction(input, output, static_cast<int>(M), static_cast<int>(N));
}

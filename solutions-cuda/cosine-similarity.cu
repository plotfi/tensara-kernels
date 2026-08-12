// cosine-similarity per row: output[i] = dot(p_i, t_i) / (||p_i|| * ||t_i||),
// for n rows of length d. A PER-ROW, MULTI-ACCUMULATOR reduction: one block per
// row reduces three sums at once (dot, ||p||^2, ||t||^2) with the shared
// SmemTreeReduce, then thread 0 writes the row's result.
#include <cuda_runtime.h>
#include "../kernel-implementation/reduction.cuh"   // SmemTreeReduce, reduce_sum, id_zero

#define BLOCK 256

__global__ void cosine_kernel(const float* __restrict__ P, const float* __restrict__ T,
                              float* __restrict__ out, int d) {
    const int row = blockIdx.x;
    const float* p = P + static_cast<size_t>(row) * d;
    const float* t = T + static_cast<size_t>(row) * d;

    float dot = 0.0f, np = 0.0f, nt = 0.0f;
    for (int j = threadIdx.x; j < d; j += BLOCK) {
        float a = p[j], b = t[j];
        dot += a * b; np += a * a; nt += b * b;
    }
    // Three block reductions (all threads reach each one uniformly).
    dot = SmemTreeReduce<BLOCK, reduce_sum, id_zero>::apply(dot);
    np  = SmemTreeReduce<BLOCK, reduce_sum, id_zero>::apply(np);
    nt  = SmemTreeReduce<BLOCK, reduce_sum, id_zero>::apply(nt);

    if (threadIdx.x == 0)
        out[row] = dot * rsqrtf(np) * rsqrtf(nt);
}

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* predictions, const float* targets, float* output,
                         size_t n, size_t d) {
    cosine_kernel<<<n, BLOCK>>>(predictions, targets, output, static_cast<int>(d));
}

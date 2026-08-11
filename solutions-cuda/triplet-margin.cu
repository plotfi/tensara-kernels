// triplet-margin loss: mean_b max(0, ||a_b - p_b|| - ||a_b - n_b|| + margin),
// Euclidean distances, B triplets of dim E. A TWO-LEVEL reduction: one block per
// triplet reduces two squared distances over E (shared SmemTreeReduce), thread 0
// forms the per-triplet loss and atomically adds loss/B to the global scalar.
#include <cuda_runtime.h>
#include "../kernel-implementation/reduction.cuh"   // SmemTreeReduce, reduce_sum, id_zero

#define BLOCK 256

__global__ void triplet_kernel(const float* __restrict__ A, const float* __restrict__ P,
                               const float* __restrict__ N, float* loss,
                               int E, float margin, float inv_B) {
    const int b = blockIdx.x;
    const float* a  = A + static_cast<size_t>(b) * E;
    const float* pp = P + static_cast<size_t>(b) * E;
    const float* nn = N + static_cast<size_t>(b) * E;

    float dp = 0.0f, dn = 0.0f;   // ||a-p||^2, ||a-n||^2
    for (int e = threadIdx.x; e < E; e += BLOCK) {
        float ap = a[e] - pp[e];
        float an = a[e] - nn[e];
        dp += ap * ap; dn += an * an;
    }
    dp = SmemTreeReduce<BLOCK, reduce_sum, id_zero>::apply(dp);
    dn = SmemTreeReduce<BLOCK, reduce_sum, id_zero>::apply(dn);

    if (threadIdx.x == 0) {
        float v = fmaxf(0.0f, sqrtf(dp) - sqrtf(dn) + margin);
        atomicAdd(loss, v * inv_B);
    }
}

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* anchor, const float* positive, const float* negative,
                         float* loss, size_t B, size_t E, float margin) {
    cudaMemset(loss, 0, sizeof(float));   // scalar accumulator
    triplet_kernel<<<B, BLOCK>>>(anchor, positive, negative, loss,
                                 static_cast<int>(E), margin, 1.0f / static_cast<float>(B));
}

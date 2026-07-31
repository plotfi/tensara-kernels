// Kernel implementation for "matrix-vector": C[i] = sum_k A[i,k] * B[k].
#include <metal_stdlib>
using namespace metal;

inline void matvec(device const float* A,
                   device const float* B,
                   device float*       C,
                   uint M, uint K, uint id) {
    if (id >= M) return;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) acc += A[id * K + k] * B[k];
    C[id] = acc;
}

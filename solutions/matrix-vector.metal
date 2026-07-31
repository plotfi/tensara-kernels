// Metal shader for "matrix-vector": C[i] = sum_k A[i,k] * B[k].  One thread per row.
#include <metal_stdlib>
using namespace metal;

kernel void solution(device const float* A [[buffer(0)]],
                     device const float* B [[buffer(1)]],
                     device float*       C [[buffer(2)]],
                     constant uint&      M [[buffer(3)]],
                     constant uint&      K [[buffer(4)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= M) return;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) acc += A[id * K + k] * B[k];
    C[id] = acc;
}

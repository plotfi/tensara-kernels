// Metal shader for "conv-1d": centered "same" cross-correlation, zero-padded.
#include <metal_stdlib>
using namespace metal;

kernel void solution(device const float* A [[buffer(0)]],
                     device const float* B [[buffer(1)]],
                     device float*       C [[buffer(2)]],
                     constant uint&      N [[buffer(3)]],
                     constant uint&      K [[buffer(4)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= N) return;
    int r = (int(K) - 1) / 2;
    float acc = 0.0f;
    for (uint j = 0; j < K; ++j) {
        int idx = int(id) + int(j) - r;
        if (idx >= 0 && idx < int(N)) acc += A[idx] * B[j];
    }
    C[id] = acc;
}

// Kernel implementation for "conv-1d": centered "same" cross-correlation.
#include <metal_stdlib>
using namespace metal;

inline void conv1d(device const float* A,
                   device const float* B,
                   device float*       C,
                   uint N, uint K, uint id) {
    if (id >= N) return;
    int r = (int(K) - 1) / 2;
    float acc = 0.0f;
    for (uint j = 0; j < K; ++j) {
        int idx = int(id) + int(j) - r;
        if (idx >= 0 && idx < int(N)) acc += A[idx] * B[j];
    }
    C[id] = acc;
}

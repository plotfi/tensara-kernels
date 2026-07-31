// Generalized convolution kernels for Metal (1D/2D/3D).
// Centered cross-correlation with zero padding, one thread per output element.
#include <metal_stdlib>
using namespace metal;

// --- 1-D ---

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

// --- 2-D ---

inline void conv2d(device const float* A,
                   device const float* B,
                   device float*       C,
                   uint H, uint W, uint Kh, uint Kw, uint id) {
    uint oh = id / W, ow = id % W;
    if (oh >= H) return;
    int rh = (int(Kh) - 1) / 2;
    int rw = (int(Kw) - 1) / 2;
    float acc = 0.0f;
    for (uint jh = 0; jh < Kh; ++jh)
        for (uint jw = 0; jw < Kw; ++jw) {
            int ih = int(oh) + int(jh) - rh;
            int iw = int(ow) + int(jw) - rw;
            if (ih >= 0 && ih < int(H) && iw >= 0 && iw < int(W))
                acc += A[ih * W + iw] * B[jh * Kw + jw];
        }
    C[id] = acc;
}

// --- 3-D (cube): K×K×K kernel on S×S×S volume ---

inline void conv3d(device const float* A,
                   device const float* B,
                   device float*       C,
                   uint S, uint K, uint id) {
    uint S2 = S * S;
    uint oz = id / S2, rem = id % S2;
    uint oy = rem / S, ox = rem % S;
    if (oz >= S) return;
    int r = (int(K) - 1) / 2;
    float acc = 0.0f;
    for (uint jz = 0; jz < K; ++jz)
        for (uint jy = 0; jy < K; ++jy)
            for (uint jx = 0; jx < K; ++jx) {
                int iz = int(oz) + int(jz) - r;
                int iy = int(oy) + int(jy) - r;
                int ix = int(ox) + int(jx) - r;
                if (iz >= 0 && iz < int(S) && iy >= 0 && iy < int(S) && ix >= 0 && ix < int(S))
                    acc += A[(iz * S + iy) * S + ix] * B[(jz * K + jy) * K + jx];
            }
    C[id] = acc;
}

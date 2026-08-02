#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>

// Fused conv1d -> maxpool1d  (Case-2 producer->producer fusion).
//
//   producer: same-length centered conv,  conv[c] = sum_j in[c+j-rc] * w[j],
//             rc = (K-1)/2, out-of-range taps treated as zero.
//   consumer: max pool over conv[] with (ks, stride, pad, dilation).
//
// Unlike an epilogue fuse, the consumer reads a *window* of producer outputs,
// so there is no single value to inline. Instead each block stages the conv
// outputs its pooled-output tile needs into shared memory, __syncthreads(),
// then reduces the pool windows straight out of shared memory -- so the conv
// signal is never written to or read back from global memory. The only global
// traffic is: read `in`/`w`, write the pooled `out`.

inline int cmp1d_out_len(int N, int ks, int pad, int dil, int stride) {
    return (N + 2 * pad - dil * (ks - 1) - 1) / stride + 1;
}

// A block owns POOL_TILE consecutive pooled outputs. The conv outputs it needs
// span [cLo, cHi):  c = o*stride + m*dil - pad  for o in the tile, m in [0,ks).
// That span is (POOL_TILE-1)*stride + (ks-1)*dil + 1 elements, staged in smem.
template <int POOL_TILE, int BLOCK>
__global__ void conv1d_maxpool1d_kernel(const float* __restrict__ in,
                                        const float* __restrict__ w,
                                        float* __restrict__ out,
                                        int N, int K, int ks, int stride, int pad, int dil,
                                        int Lout) {
    const int rc  = (K - 1) / 2;
    const int o0  = blockIdx.x * POOL_TILE;
    const int cLo = o0 * stride - pad;                                   // first conv idx needed
    const int span = (POOL_TILE - 1) * stride + (ks - 1) * dil + 1;      // conv outputs to stage

    extern __shared__ float sConv[];   // span floats

    // PRODUCER: compute the needed conv outputs into shared memory. A conv index
    // outside [0,N) is a padded pool tap -> store -inf so the max ignores it.
    for (int t = threadIdx.x; t < span; t += BLOCK) {
        int c = cLo + t;
        if (c < 0 || c >= N) { sConv[t] = -CUDART_INF_F; continue; }
        float acc = 0.0f;
        for (int j = 0; j < K; ++j) {
            int xi = c + j - rc;
            if (xi >= 0 && xi < N) acc += in[xi] * w[j];
        }
        sConv[t] = acc;
    }
    __syncthreads();

    // CONSUMER: max-pool the staged conv outputs. c - cLo is always in [0,span)
    // by construction, so no bounds check on the shared read is needed.
    for (int p = threadIdx.x; p < POOL_TILE; p += BLOCK) {
        int o = o0 + p;
        if (o >= Lout) break;
        float m = -CUDART_INF_F;
        for (int k = 0; k < ks; ++k) {
            int c = o * stride + k * dil - pad;
            m = fmaxf(m, sConv[c - cLo]);
        }
        out[o] = m;
    }
}

template <int POOL_TILE = 128, int BLOCK = 128>
inline void launch_conv1d_maxpool1d(const float* in, const float* w, float* out,
                                    int N, int K, int ks, int stride, int pad, int dil) {
    int Lout = cmp1d_out_len(N, ks, pad, dil, stride);
    if (Lout <= 0) return;
    int grid = (Lout + POOL_TILE - 1) / POOL_TILE;
    int span = (POOL_TILE - 1) * stride + (ks - 1) * dil + 1;
    size_t smem = static_cast<size_t>(span) * sizeof(float);
    conv1d_maxpool1d_kernel<POOL_TILE, BLOCK>
        <<<grid, BLOCK, smem>>>(in, w, out, N, K, ks, stride, pad, dil, Lout);
}

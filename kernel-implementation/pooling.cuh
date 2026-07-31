#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>

// ---------------------------------------------------------------------------
// Reduce operations for pooling: AvgPoolOp divides by window size,
// MaxPoolOp takes the maximum.  Each op defines init(), combine(), and
// finalize() so the kernel body is identical for both.
// ---------------------------------------------------------------------------

struct AvgPoolOp {
    static __device__ float init()                          { return 0.0f; }
    static __device__ float combine(float a, float b)       { return a + b; }
    static __device__ float finalize(float acc, int count)  { return acc / static_cast<float>(count); }
};

struct MaxPoolOp {
    static __device__ float init()                          { return -CUDART_INF_F; }
    static __device__ float combine(float a, float b)       { return fmaxf(a, b); }
    static __device__ float finalize(float acc, int)        { return acc; }
};

// ---------------------------------------------------------------------------
// 1-D pooling   (avg-pool-1d, max-pool-1d)
// ---------------------------------------------------------------------------

template <typename Op>
__global__ void pool1d_kernel(const float* input, float* output,
                              int ks, int stride, int pad, int dilation,
                              int H, int Hout) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= Hout) return;

    float acc = Op::init();
    for (int m = 0; m < ks; ++m) {
        int ri = stride * i + m * dilation - pad;
        if (ri >= 0 && ri < H)
            acc = Op::combine(acc, input[ri]);
    }
    output[i] = Op::finalize(acc, ks);
}

// ---------------------------------------------------------------------------
// 2-D pooling   (avg-pool-2d, max-pool-2d)
// ---------------------------------------------------------------------------

template <typename Op>
__global__ void pool2d_kernel(const float* input, float* output,
                              int ks, int stride, int pad, int dilation,
                              int H, int W, int Hout, int Wout) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int oh = idx / Wout, ow = idx % Wout;
    if (oh >= Hout) return;

    float acc = Op::init();
    for (int mh = 0; mh < ks; ++mh)
        for (int mw = 0; mw < ks; ++mw) {
            int rh = stride * oh + mh * dilation - pad;
            int rw = stride * ow + mw * dilation - pad;
            if (rh >= 0 && rh < H && rw >= 0 && rw < W)
                acc = Op::combine(acc, input[rh * W + rw]);
        }
    output[idx] = Op::finalize(acc, ks * ks);
}

// ---------------------------------------------------------------------------
// 3-D pooling   (avg-pool-3d, max-pool-3d)
// ---------------------------------------------------------------------------

template <typename Op>
__global__ void pool3d_kernel(const float* input, float* output,
                              int ks, int stride, int pad, int dilation,
                              int H, int W, int D,
                              int Hout, int Wout, int Dout) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int WDout = Wout * Dout;
    int oh = idx / WDout;
    int ow = (idx % WDout) / Dout;
    int od = idx % Dout;
    if (oh >= Hout) return;

    float acc = Op::init();
    for (int mh = 0; mh < ks; ++mh)
        for (int mw = 0; mw < ks; ++mw)
            for (int md = 0; md < ks; ++md) {
                int rh = stride * oh + mh * dilation - pad;
                int rw = stride * ow + mw * dilation - pad;
                int rd = stride * od + md * dilation - pad;
                if (rh >= 0 && rh < H && rw >= 0 && rw < W && rd >= 0 && rd < D)
                    acc = Op::combine(acc, input[(rh * W + rw) * D + rd]);
            }
    output[idx] = Op::finalize(acc, ks * ks * ks);
}

// ---------------------------------------------------------------------------
// Output-size helper (same formula for every dimension).
// ---------------------------------------------------------------------------

inline int pool_out_size(int in, int ks, int pad, int dilation, int stride) {
    return (in + 2 * pad - dilation * (ks - 1) - 1) / stride + 1;
}

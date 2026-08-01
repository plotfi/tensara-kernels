#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>

// Pooling generalized over the *scalar reduce op* only (not dimensionality).
// AvgPoolOp / MaxPoolOp each define init()/combine()/finalize() so avg-pool
// and max-pool share one kernel body. The 1-D -> 2-D/3-D expansion is left to
// the caller: add pool2d_kernel/pool3d_kernel yourself when you need them.

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

// 1-D pooling: one output element per thread. `finalize` divides by the window
// size `ks` for AvgPoolOp (count_include_pad semantics); MaxPoolOp ignores it.
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

// Output length along one axis (same formula every dimension).
inline int pool_out_size(int in, int ks, int pad, int dilation, int stride) {
    return (in + 2 * pad - dilation * (ks - 1) - 1) / stride + 1;
}

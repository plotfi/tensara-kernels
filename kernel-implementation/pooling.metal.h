// Pooling generalized over the scalar reduce op only (Metal side). Mirrors
// pooling.cuh: AvgPoolOp/MaxPoolOp share one 1-D kernel body. Add pool2d/pool3d
// yourself when you need higher dimensionality.
#include <metal_stdlib>
using namespace metal;

struct AvgPoolOp {
    static float init()                         { return 0.0f; }
    static float combine(float a, float b)      { return a + b; }
    static float finalize(float acc, int count) { return acc / float(count); }
};

struct MaxPoolOp {
    static float init()                         { return -INFINITY; }
    static float combine(float a, float b)      { return max(a, b); }
    static float finalize(float acc, int)       { return acc; }
};

template <typename Op>
inline void pool1d(device const float* in, device float* out,
                   int ks, int stride, int pad, int dilation, int H, uint id) {
    int Hout = (H + 2 * pad - dilation * (ks - 1) - 1) / stride + 1;
    if (int(id) >= Hout) return;
    float acc = Op::init();
    for (int m = 0; m < ks; ++m) {
        int ri = stride * int(id) + m * dilation - pad;
        if (ri >= 0 && ri < H) acc = Op::combine(acc, in[ri]);
    }
    out[id] = Op::finalize(acc, ks);
}

// Generalized pooling kernels for Metal (1D/2D/3D, avg and max).
// Each .metal shader #includes this and calls the appropriate variant.
#include <metal_stdlib>
using namespace metal;

struct AvgPoolOp {
    static float init()                        { return 0.0f; }
    static float combine(float a, float b)     { return a + b; }
    static float finalize(float acc, int count) { return acc / float(count); }
};

struct MaxPoolOp {
    static float init()                        { return -INFINITY; }
    static float combine(float a, float b)     { return max(a, b); }
    static float finalize(float acc, int)      { return acc; }
};

// --- 1-D ---

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

// --- 2-D ---

template <typename Op>
inline void pool2d(device const float* in, device float* out,
                   int ks, int stride, int pad, int dilation,
                   int H, int W, uint id) {
    int Hout = (H + 2 * pad - dilation * (ks - 1) - 1) / stride + 1;
    int Wout = (W + 2 * pad - dilation * (ks - 1) - 1) / stride + 1;
    int oh = int(id) / Wout, ow = int(id) % Wout;
    if (oh >= Hout) return;
    float acc = Op::init();
    for (int mh = 0; mh < ks; ++mh)
        for (int mw = 0; mw < ks; ++mw) {
            int rh = stride * oh + mh * dilation - pad;
            int rw = stride * ow + mw * dilation - pad;
            if (rh >= 0 && rh < H && rw >= 0 && rw < W)
                acc = Op::combine(acc, in[rh * W + rw]);
        }
    out[id] = Op::finalize(acc, ks * ks);
}

// --- 3-D ---

template <typename Op>
inline void pool3d(device const float* in, device float* out,
                   int ks, int stride, int pad, int dilation,
                   int H, int W, int D, uint id) {
    int Hout = (H + 2 * pad - dilation * (ks - 1) - 1) / stride + 1;
    int Wout = (W + 2 * pad - dilation * (ks - 1) - 1) / stride + 1;
    int Dout = (D + 2 * pad - dilation * (ks - 1) - 1) / stride + 1;
    int WDout = Wout * Dout;
    int oh = int(id) / WDout;
    int ow = (int(id) % WDout) / Dout;
    int od = int(id) % Dout;
    if (oh >= Hout) return;
    float acc = Op::init();
    for (int mh = 0; mh < ks; ++mh)
        for (int mw = 0; mw < ks; ++mw)
            for (int md = 0; md < ks; ++md) {
                int rh = stride * oh + mh * dilation - pad;
                int rw = stride * ow + mw * dilation - pad;
                int rd = stride * od + md * dilation - pad;
                if (rh >= 0 && rh < H && rw >= 0 && rw < W && rd >= 0 && rd < D)
                    acc = Op::combine(acc, in[(rh * W + rw) * D + rd]);
            }
    out[id] = Op::finalize(acc, ks * ks * ks);
}

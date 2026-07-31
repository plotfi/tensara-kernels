// Kernel implementation for "avg-pool-1d" (count_include_pad, divide by kernel_size).
#include <metal_stdlib>
using namespace metal;

inline void avgpool1d(device const float* in,
                      device float*       out,
                      int ks, int stride, int pad, int H, uint id) {
    int Hout = (H + 2 * pad - ks) / stride + 1;
    if (int(id) >= Hout) return;
    float sum = 0.0f;
    for (int m = 0; m < ks; ++m) {
        int ri = stride * int(id) + m - pad;
        if (ri >= 0 && ri < H) sum += in[ri];
    }
    out[id] = sum / float(ks);
}

// Metal shader for "avg-pool-1d" (count_include_pad, divide by kernel_size).
#include <metal_stdlib>
using namespace metal;

kernel void solution(device const float* in  [[buffer(0)]],
                     device float*       out [[buffer(1)]],
                     constant int&       ks     [[buffer(2)]],
                     constant int&       stride [[buffer(3)]],
                     constant int&       pad    [[buffer(4)]],
                     constant int&       H      [[buffer(5)]],
                     uint id [[thread_position_in_grid]]) {
    int Hout = (H + 2 * pad - ks) / stride + 1;
    if (int(id) >= Hout) return;
    float sum = 0.0f;
    for (int m = 0; m < ks; ++m) {
        int ri = stride * int(id) + m - pad;
        if (ri >= 0 && ri < H) sum += in[ri];
    }
    out[id] = sum / float(ks);
}

// Metal shader for "swish" (elementwise activation).
#include <metal_stdlib>
using namespace metal;

kernel void solution(device const float* in [[buffer(0)]],
                     device float*       out [[buffer(1)]],
                     constant uint&      n   [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= n) return;
    float x = in[id];
    out[id] = x / (1.0f + exp(-x));
}

// Metal shader for "hard-sigmoid" (elementwise activation).
#include <metal_stdlib>
using namespace metal;

kernel void solution(device const float* in [[buffer(0)]],
                     device float*       out [[buffer(1)]],
                     constant uint&      n   [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= n) return;
    float x = in[id];
    out[id] = (x <= -3.0f) ? 0.0f : ((x >= 3.0f) ? 1.0f : (x + 3.0f) / 6.0f);
}

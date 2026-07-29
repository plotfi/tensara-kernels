// Metal shader for "elu" (elementwise; takes an alpha parameter).
#include <metal_stdlib>
using namespace metal;

kernel void solution(device const float* in [[buffer(0)]],
                     device float*       out   [[buffer(1)]],
                     constant uint&      n     [[buffer(2)]],
                     constant float&     alpha [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= n) return;
    float x = in[id];
    out[id] = (x > 0.0f) ? x : alpha * (exp(x) - 1.0f);
}

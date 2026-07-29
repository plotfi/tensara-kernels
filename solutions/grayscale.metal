// Metal shader for "grayscale": HWC RGB -> Rec.601 luminance.
#include <metal_stdlib>
using namespace metal;

kernel void solution(device const float* rgb  [[buffer(0)]],
                     device float*       gray [[buffer(1)]],
                     constant uint&      n    [[buffer(2)]],  // height*width
                     uint id [[thread_position_in_grid]]) {
    if (id >= n) return;
    uint b = id * 3;
    gray[id] = 0.299f * rgb[b] + 0.587f * rgb[b + 1] + 0.114f * rgb[b + 2];
}

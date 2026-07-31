// Kernel implementation for "grayscale": HWC RGB -> Rec.601 luminance.
#include <metal_stdlib>
using namespace metal;

inline void grayscale(device const float* rgb,
                      device float*       gray,
                      uint n, uint id) {
    if (id >= n) return;
    uint b = id * 3;
    gray[id] = 0.299f * rgb[b] + 0.587f * rgb[b + 1] + 0.114f * rgb[b + 2];
}

// Metal shader for "avg-pool-1d".
#include "avg-pool-1d.metal.h"

kernel void solution(device const float* in  [[buffer(0)]],
                     device float*       out [[buffer(1)]],
                     constant int&       ks     [[buffer(2)]],
                     constant int&       stride [[buffer(3)]],
                     constant int&       pad    [[buffer(4)]],
                     constant int&       H      [[buffer(5)]],
                     uint id [[thread_position_in_grid]]) {
    avgpool1d(in, out, ks, stride, pad, H, id);
}

// Metal shader for "max-pool-1d" -- generalized pooling, op = MaxPoolOp.
#include "pooling.metal.h"

kernel void solution(device const float* in       [[buffer(0)]],
                     device float*       out      [[buffer(1)]],
                     constant int&       ks       [[buffer(2)]],
                     constant int&       stride   [[buffer(3)]],
                     constant int&       pad      [[buffer(4)]],
                     constant int&       dilation [[buffer(5)]],
                     constant int&       H        [[buffer(6)]],
                     uint id [[thread_position_in_grid]]) {
    pool1d<MaxPoolOp>(in, out, ks, stride, pad, dilation, H, id);
}

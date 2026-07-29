// Metal shader for "mean-subtract" -- unified reduction kernel, op = MeanSubOps.
#include "reduction.metal.h"

kernel void solution(device const float* X [[buffer(0)]],
                     device float*       Y [[buffer(1)]],
                     constant uint&      D [[buffer(2)]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint row [[threadgroup_position_in_grid]],
                     uint tpg [[threads_per_threadgroup]]) {
    threadgroup float smem[256];
    reduce_row<MeanSubOps>(X, Y, D, tid, row, tpg, smem);
}

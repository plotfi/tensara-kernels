// Metal shader for "log-softmax": one threadgroup per row, tree reduction over the row.
#include <metal_stdlib>
using namespace metal;

kernel void solution(device const float* X [[buffer(0)]],
                     device float*       Y [[buffer(1)]],
                     constant uint&      D [[buffer(2)]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint row [[threadgroup_position_in_grid]],
                     uint tpg [[threads_per_threadgroup]]) {
    threadgroup float smem[256];
    const device float* rx = X + (uint)row * D;
    device float*       ry = Y + (uint)row * D;

    float acc = 0.0f;
    for (uint i = tid; i < D; i += tpg) { float v = rx[i]; acc = acc + exp(v); }
    smem[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s) { float a = smem[tid], b = smem[tid + s]; smem[tid] = a + b; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float f;
    { float s = smem[0]; f = log(s); }
    for (uint i = tid; i < D; i += tpg) { float x = rx[i]; ry[i] = x - f; }
}

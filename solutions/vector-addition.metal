// Metal shader for "vector-addition": C[i] = A[i] + B[i].
// Buffers 0..2 are the tensors; buffer 3 is the element count (scalar).
#include <metal_stdlib>
using namespace metal;

kernel void solution(device const float* a [[buffer(0)]],
                     device const float* b [[buffer(1)]],
                     device float*       c [[buffer(2)]],
                     constant uint&      n [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    if (id < n) c[id] = a[id] + b[id];
}

// Metal shader for "vector-addition".
#include "vector-addition.metal.h"

kernel void solution(device const float* a [[buffer(0)]],
                     device const float* b [[buffer(1)]],
                     device float*       c [[buffer(2)]],
                     constant uint&      n [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    vector_add(a, b, c, n, id);
}

// Metal shader for "conv-1d".
#include "convolution.metal.h"

kernel void solution(device const float* A [[buffer(0)]],
                     device const float* B [[buffer(1)]],
                     device float*       C [[buffer(2)]],
                     constant uint&      N [[buffer(3)]],
                     constant uint&      K [[buffer(4)]],
                     uint id [[thread_position_in_grid]]) {
    conv1d(A, B, C, N, K, id);
}

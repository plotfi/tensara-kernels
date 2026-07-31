// Metal shader for "selu" -- unified activation kernel, op = Selu.
#include "activation.metal.h"

kernel void solution(device const float* A [[buffer(0)]],
                     device float*       C [[buffer(1)]],
                     constant uint&      n [[buffer(2)]],
                     constant float&     alpha [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    activation_x8<Selu>(A, C, n, alpha, gid);
}

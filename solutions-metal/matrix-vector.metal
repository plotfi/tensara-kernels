// Metal shader for "matrix-vector".
#include "matrix-vector.metal.h"

kernel void solution(device const float* A [[buffer(0)]],
                     device const float* B [[buffer(1)]],
                     device float*       C [[buffer(2)]],
                     constant uint&      M [[buffer(3)]],
                     constant uint&      K [[buffer(4)]],
                     uint id [[thread_position_in_grid]]) {
    matvec(A, B, C, M, K, id);
}

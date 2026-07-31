// Metal shader for "grayscale".
#include "grayscale.metal.h"

kernel void solution(device const float* rgb  [[buffer(0)]],
                     device float*       gray [[buffer(1)]],
                     constant uint&      n    [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    grayscale(rgb, gray, n, id);
}

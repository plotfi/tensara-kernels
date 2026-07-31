// Kernel implementation for "vector-addition": C[i] = A[i] + B[i].
#include <metal_stdlib>
using namespace metal;

inline void vector_add(device const float* a,
                       device const float* b,
                       device float*       c,
                       uint n, uint id) {
    if (id < n) c[id] = a[id] + b[id];
}

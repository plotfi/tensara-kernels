// Metal solution wrapper for "vector-addition".
#include "../tensor-lib/tensor.cuh"

METAL_KERNEL("vector-addition", R"(
// Metal shader for "vector-addition".
#include "vector-addition.metal.h"

kernel void solution(device const float* a [[buffer(0)]],
                     device const float* b [[buffer(1)]],
                     device float*       c [[buffer(2)]],
                     constant uint&      n [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    vector_add(a, b, c, n, id);
}
)");

extern "C" void solution(const float* d_input1, const float* d_input2, float* d_output, size_t n) {
    auto pso = metal_pso();
    uint32_t N = static_cast<uint32_t>(n);
    tensor::dispatch(pso, { tensor::buf(d_input1), tensor::buf(d_input2), tensor::buf(d_output) },
                      { tensor::arg(N) }, n);
}

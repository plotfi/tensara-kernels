// Metal solution wrapper for "matrix-vector".
#include "../tensor-lib/tensor.cuh"

METAL_KERNEL("matrix-vector", R"(
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
)");

extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t k) {
    auto pso = metal_pso();
    uint32_t M = static_cast<uint32_t>(m), K = static_cast<uint32_t>(k);
    tensor::dispatch(pso, { tensor::buf(input_a), tensor::buf(input_b), tensor::buf(output_c) },
                      { tensor::arg(M), tensor::arg(K) }, m);
}

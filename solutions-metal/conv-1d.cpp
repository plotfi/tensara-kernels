// Metal solution wrapper for "conv-1d".
#include "../tensor-lib/tensor.cuh"

METAL_KERNEL("conv-1d", R"(
// Metal shader for "conv-1d".
#include "conv-1d.metal.h"

kernel void solution(device const float* A [[buffer(0)]],
                     device const float* B [[buffer(1)]],
                     device float*       C [[buffer(2)]],
                     constant uint&      N [[buffer(3)]],
                     constant uint&      K [[buffer(4)]],
                     uint id [[thread_position_in_grid]]) {
    conv1d(A, B, C, N, K, id);
}
)");

extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    auto pso = metal_pso();
    uint32_t Nn = static_cast<uint32_t>(N), Kk = static_cast<uint32_t>(K);
    tensor::dispatch(pso, { tensor::buf(A), tensor::buf(B), tensor::buf(C) },
                      { tensor::arg(Nn), tensor::arg(Kk) }, N);
}

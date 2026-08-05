// Metal solution wrapper for "elu" (unified activation kernel).
#include "../tensor-lib/tensor.cuh"

METAL_KERNEL("elu", R"(
// Metal shader for "elu" -- unified activation kernel, op = Elu.
#include "activation.metal.h"

kernel void solution(device const float* A [[buffer(0)]],
                     device float*       C [[buffer(1)]],
                     constant uint&      n [[buffer(2)]],
                     constant float&     alpha [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    activation_x8<Elu>(A, C, n, alpha, gid);
}
)");

extern "C" void solution(const float* input, float* output, size_t n, size_t m, float alpha) {
    auto pso = metal_pso();
    uint32_t N = static_cast<uint32_t>(n * m);
    size_t threads = (N + 7u) / 8u;
    tensor::dispatch(pso, { tensor::buf(input), tensor::buf(output) },
                      { tensor::arg(N), tensor::arg(alpha) }, threads);
}

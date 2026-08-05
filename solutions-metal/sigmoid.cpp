// Metal solution wrapper for "sigmoid" (unified activation kernel, 8 elems/thread).
#include "../tensor-lib/tensor.cuh"

METAL_KERNEL("sigmoid", R"(
// Metal shader for "sigmoid" -- unified activation kernel, op = Sigmoid.
#include "activation.metal.h"

kernel void solution(device const float* A [[buffer(0)]],
                     device float*       C [[buffer(1)]],
                     constant uint&      n [[buffer(2)]],
                     constant float&     alpha [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    activation_x8<Sigmoid>(A, C, n, alpha, gid);
}
)");

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    auto pso = metal_pso();
    uint32_t N = static_cast<uint32_t>(n * m);
    float alpha = 0.0f;
    size_t threads = (N + 7u) / 8u;
    tensor::dispatch(pso, { tensor::buf(input), tensor::buf(output) },
                      { tensor::arg(N), tensor::arg(alpha) }, threads);
}

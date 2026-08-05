// Metal solution wrapper for "threshold" (unified activation kernel, 8 elems/thread).
// Mirrors the CUDA solution: alpha carries the threshold value.
#include "../tensor-lib/tensor.cuh"

METAL_KERNEL("threshold", R"(
// Metal shader for "threshold" -- unified activation kernel, op = Threshold.
#include "activation.metal.h"

kernel void solution(device const float* A [[buffer(0)]],
                     device float*       C [[buffer(1)]],
                     constant uint&      n [[buffer(2)]],
                     constant float&     alpha [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    activation_x8<Threshold>(A, C, n, alpha, gid);
}
)");

extern "C" void solution(const float* input_image, float threshold_value, float* output_image, size_t height, size_t width) {
    auto pso = metal_pso();
    uint32_t N = static_cast<uint32_t>(height * width);
    float alpha = threshold_value;
    size_t threads = (N + 7u) / 8u;
    tensor::dispatch(pso, { tensor::buf(input_image), tensor::buf(output_image) },
                      { tensor::arg(N), tensor::arg(alpha) }, threads);
}

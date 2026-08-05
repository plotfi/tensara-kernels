// Metal solution wrapper for "grayscale".
#include "../tensor-lib/tensor.cuh"

METAL_KERNEL("grayscale", R"(
// Metal shader for "grayscale".
#include "grayscale.metal.h"

kernel void solution(device const float* rgb  [[buffer(0)]],
                     device float*       gray [[buffer(1)]],
                     constant uint&      n    [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    grayscale(rgb, gray, n, id);
}
)");

extern "C" void solution(const float* rgb_image, float* grayscale_output, size_t height, size_t width, size_t channels) {
    auto pso = metal_pso();
    uint32_t N = static_cast<uint32_t>(height * width);
    tensor::dispatch(pso, { tensor::buf(rgb_image), tensor::buf(grayscale_output) },
                      { tensor::arg(N) }, height * width);
}

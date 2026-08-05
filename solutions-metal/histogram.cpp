// Metal solution wrapper stub for "histogram" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

METAL_KERNEL("histogram", R"(
// Metal shader stub for "histogram" (unimplemented). Fill in like the CUDA
// solution once the algorithm exists; bind the same buffers/scalars in the
// harness's Metal branch and dispatch over the output.
#include <metal_stdlib>
using namespace metal;

kernel void solution(uint id [[thread_position_in_grid]]) {
    // TODO: implement histogram
}
)");

extern "C" void solution(const float* image, int num_bins, float* histogram, size_t height, size_t width) {
    auto pso = metal_pso();
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

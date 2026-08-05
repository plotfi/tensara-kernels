// Metal solution wrapper stub for "conv2d-relu-hardswish" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

METAL_KERNEL("conv2d-relu-hardswish", R"(
// Metal shader stub for "conv2d-relu-hardswish" (unimplemented). Fill in like the CUDA
// solution once the algorithm exists; bind the same buffers/scalars in the
// harness's Metal branch and dispatch over the output.
#include <metal_stdlib>
using namespace metal;

kernel void solution(uint id [[thread_position_in_grid]]) {
    // TODO: implement conv2d-relu-hardswish
}
)");

extern "C" void solution(const float* image, const float* kernel, float* output, size_t H, size_t W, size_t Kh, size_t Kw) {
    auto pso = metal_pso();
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

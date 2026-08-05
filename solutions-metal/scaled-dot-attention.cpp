// Metal solution wrapper stub for "scaled-dot-attention" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

METAL_KERNEL("scaled-dot-attention", R"(
// Metal shader stub for "scaled-dot-attention" (unimplemented). Fill in like the CUDA
// solution once the algorithm exists; bind the same buffers/scalars in the
// harness's Metal branch and dispatch over the output.
#include <metal_stdlib>
using namespace metal;

kernel void solution(uint id [[thread_position_in_grid]]) {
    // TODO: implement scaled-dot-attention
}
)");

extern "C" void solution(const float* Q, const float* K, const float* V, float* output, size_t B, size_t H, size_t S, size_t E) {
    auto pso = metal_pso();
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

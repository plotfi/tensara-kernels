// Metal solution wrapper stub for "cosine-similarity" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

METAL_KERNEL("cosine-similarity", R"(
// Metal shader stub for "cosine-similarity" (unimplemented). Fill in like the CUDA
// solution once the algorithm exists; bind the same buffers/scalars in the
// harness's Metal branch and dispatch over the output.
#include <metal_stdlib>
using namespace metal;

kernel void solution(uint id [[thread_position_in_grid]]) {
    // TODO: implement cosine-similarity
}
)");

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n, size_t d) {
    auto pso = metal_pso();
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

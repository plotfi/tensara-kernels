// Metal solution wrapper stub for "vector-multiply-ff" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

METAL_KERNEL("vector-multiply-ff", R"(
// Metal shader stub for "vector-multiply-ff" (unimplemented). Fill in like the CUDA
// solution once the algorithm exists; bind the same buffers/scalars in the
// harness's Metal branch and dispatch over the output.
#include <metal_stdlib>
using namespace metal;

kernel void solution(uint id [[thread_position_in_grid]]) {
    // TODO: implement vector-multiply-ff
}
)");

extern "C" void solution(const uint32_t* d_input1, const uint32_t* d_input2, uint32_t* d_output, size_t n) {
    auto pso = metal_pso();
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}

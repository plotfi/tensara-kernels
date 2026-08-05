// Metal solution wrapper for "avg-pool-1d".
#include "../tensor-lib/tensor.cuh"

METAL_KERNEL("avg-pool-1d", R"(
// Metal shader for "avg-pool-1d" -- generalized pooling, op = AvgPoolOp.
#include "pooling.metal.h"

kernel void solution(device const float* in  [[buffer(0)]],
                     device float*       out [[buffer(1)]],
                     constant int&       ks     [[buffer(2)]],
                     constant int&       stride [[buffer(3)]],
                     constant int&       pad    [[buffer(4)]],
                     constant int&       H      [[buffer(5)]],
                     uint id [[thread_position_in_grid]]) {
    pool1d<AvgPoolOp>(in, out, ks, stride, pad, 1, H, id);
}
)");

extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H) {
    auto pso = metal_pso();
    int Hi = static_cast<int>(H);
    int Hout = (Hi + 2 * padding - kernel_size) / stride + 1;
    tensor::dispatch(pso, { tensor::buf(input), tensor::buf(output) },
                      { tensor::arg(kernel_size), tensor::arg(stride), tensor::arg(padding), tensor::arg(Hi) },
                      static_cast<size_t>(Hout));
}

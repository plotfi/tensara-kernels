// Metal solution wrapper for "max-pool-1d". Mirrors the CUDA solution: dispatch
// one thread per output element over the generalized pool1d<MaxPoolOp> shader.
#include "../tensor-lib/tensor.cuh"

METAL_KERNEL("max-pool-1d", R"(
// Metal shader for "max-pool-1d" -- generalized pooling, op = MaxPoolOp.
#include "pooling.metal.h"

kernel void solution(device const float* in       [[buffer(0)]],
                     device float*       out      [[buffer(1)]],
                     constant int&       ks       [[buffer(2)]],
                     constant int&       stride   [[buffer(3)]],
                     constant int&       pad      [[buffer(4)]],
                     constant int&       dilation [[buffer(5)]],
                     constant int&       H        [[buffer(6)]],
                     uint id [[thread_position_in_grid]]) {
    pool1d<MaxPoolOp>(in, out, ks, stride, pad, dilation, H, id);
}
)");

extern "C" void solution(const float* input, int kernel_size, int stride, int padding, int dilation, float* output, size_t H) {
    auto pso = metal_pso();
    int Hi = static_cast<int>(H);
    int Hout = (Hi + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    tensor::dispatch(pso, { tensor::buf(input), tensor::buf(output) },
                      { tensor::arg(kernel_size), tensor::arg(stride), tensor::arg(padding),
                        tensor::arg(dilation), tensor::arg(Hi) },
                      static_cast<size_t>(Hout));
}

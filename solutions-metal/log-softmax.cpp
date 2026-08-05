// Metal solution wrapper for "log-softmax".
#include "../tensor-lib/tensor.cuh"

METAL_KERNEL("log-softmax", R"(
// Metal shader for "log-softmax" -- unified reduction kernel, op = LogSoftmaxOps.
#include "reduction.metal.h"

kernel void solution(device const float* X [[buffer(0)]],
                     device float*       Y [[buffer(1)]],
                     constant uint&      D [[buffer(2)]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint row [[threadgroup_position_in_grid]],
                     uint tpg [[threads_per_threadgroup]]) {
    threadgroup float smem[256];
    reduce_row<LogSoftmaxOps>(X, Y, D, tid, row, tpg, smem);
}
)");

extern "C" void solution(const float* input, float* output, size_t M, size_t N) {
    auto pso = metal_pso();
    uint32_t Dv = static_cast<uint32_t>(N);
    const size_t tpg = 256;
    tensor::dispatch(pso, { tensor::buf(input), tensor::buf(output) }, { tensor::arg(Dv) }, M * tpg, tpg);
}

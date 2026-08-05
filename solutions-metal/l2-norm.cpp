// Metal solution wrapper for "l2-norm".
#include "../tensor-lib/tensor.cuh"

METAL_KERNEL("l2-norm", R"(
// Metal shader for "l2-norm" -- unified reduction kernel, op = L2NormOps.
#include "reduction.metal.h"

kernel void solution(device const float* X [[buffer(0)]],
                     device float*       Y [[buffer(1)]],
                     constant uint&      D [[buffer(2)]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint row [[threadgroup_position_in_grid]],
                     uint tpg [[threads_per_threadgroup]]) {
    threadgroup float smem[256];
    reduce_row<L2NormOps>(X, Y, D, tid, row, tpg, smem);
}
)");

extern "C" void solution(const float* X, float* Y, size_t B, size_t D) {
    auto pso = metal_pso();
    uint32_t Dv = static_cast<uint32_t>(D);
    const size_t tpg = 256;
    tensor::dispatch(pso, { tensor::buf(X), tensor::buf(Y) }, { tensor::arg(Dv) }, B * tpg, tpg);
}

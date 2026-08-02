#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>
#include "scalar-ops.cuh"   // shared scalar-op functors (Relu, Swish, ...)

// Templated on a scalar-op functor ACT (from scalar-ops.cuh) and BLOCK_SIZE so
// each activation gets its own specialized kernel with a tuned launch config.
// ACT::apply(x, alpha) is __forceinline__, so it is inlined at every call.
//
// __restrict__ tells the compiler A and C do not alias, unlocking better
// instruction scheduling (loads and stores can be reordered freely, all loads
// can be issued up front to hide memory latency).
//
// n is int (32 bit) instead of size_t (64 bit). 32 bit index math lowers to
// simpler SASS: address calcs and loop increments avoid the 64 bit multiply
// and carry chain that size_t forces. Meaningful savings on tight loops.
template <typename ACT, int BLOCK_SIZE>
__global__ void activation_kernelx8(const float* __restrict__ A,
                                    float* __restrict__ C,
                                    int n,
                                    float alpha) {
    // Each thread handles 8 elements. B200 tensor cores and load/store units
    // can move 256 bits at a time (two float4s back to back), so 8 floats per
    // thread lets the compiler keep two 128 bit loads in flight simultaneously,
    // maxing out memory-level parallelism per thread.
    int base = (threadIdx.x + blockIdx.x * blockDim.x) * 8;

    // Scalar tail for the last partial group of 8. Uses the same ACT::apply
    // so the tail matches the vectorized path exactly.
    if (base + 7 >= n) {
        for (int j = base; j < n; ++j) {
            C[j] = ACT::apply(A[j], alpha);
        }
        return;
    }

    // Both float4 loads issued back to back. The compiler keeps both
    // LDG.E.128 instructions in flight without waiting on either, giving the
    // memory subsystem two outstanding requests per thread to hide HBM latency.
    float4 x0 = *reinterpret_cast<const float4*>(A + base);
    float4 x1 = *reinterpret_cast<const float4*>(A + base + 4);

    // 8 independent activation evaluations per thread. With __forceinline__ on
    // ACT::apply, these lower to 8 straight-line SFU or ALU sequences with no
    // register spills and full ILP exposure.
    x0.x = ACT::apply(x0.x, alpha); x0.y = ACT::apply(x0.y, alpha);
    x0.z = ACT::apply(x0.z, alpha); x0.w = ACT::apply(x0.w, alpha);
    x1.x = ACT::apply(x1.x, alpha); x1.y = ACT::apply(x1.y, alpha);
    x1.z = ACT::apply(x1.z, alpha); x1.w = ACT::apply(x1.w, alpha);

    // Two 128 bit stores, mirroring the load pattern. STG.E.128 x 2.
    *reinterpret_cast<float4*>(C + base    ) = x0;
    *reinterpret_cast<float4*>(C + base + 4) = x1;
}

// Host-side launcher. Marked __forceinline__ so it collapses into solution()
// at the call site. No perf impact worth measuring, just cleaner codegen.
// ACT is a scalar-op functor from scalar-ops.cuh, e.g. multi_solution<Relu, 512>.
template <typename ACT, int BLOCK_SIZE>
__forceinline__ void multi_solution(const float* input, float* output, size_t n, size_t m,
                                    float alpha) {
    // Cast to int for the same 32 bit arithmetic reason as the kernel.
    int N = static_cast<int>(n * m);
    if (N == 0) return;

    // Each thread does 8 elements, so we need ceil(N/8) threads total.
    int threads_needed = (N + 7) / 8;
    const int grid = (threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;

    activation_kernelx8<ACT, BLOCK_SIZE>
        <<<grid, BLOCK_SIZE>>>(input, output, N, alpha);
}

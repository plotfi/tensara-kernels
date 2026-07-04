#pragma once
#include <cuda_runtime.h>

// Generic one-block-per-row reduction + elementwise-writeback kernel.
//
// A row is reduced to a single scalar, that scalar is finalized, then every
// element of the row is rewritten as a function of (element, finalized scalar).
// An instantiation defines these macros before including this header:
//
//   REDUCE_INIT        identity value for the reduction   (default 0.0f)
//   COMBINE(A, B)      reduction combine operator         (default sum)
//   MAP_OP(X)          map an element before reducing     (required)
//   FINALIZE(ACC, D)   finalize the reduced accumulator   (required)
//   WRITE_OP(X, ACC)   per-element output transform       (required)
//
// Assumes the row length D is a multiple of 4 (float4 vectorized) and
// BLOCK_SIZE is a power of two.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#ifndef REDUCE_INIT
#define REDUCE_INIT 0.0f
#endif

#ifndef COMBINE
#define COMBINE(A, B) ((A) + (B))
#endif

__global__ void reduction_kernel(const float* __restrict__ X,
                                 float* __restrict__ Y, int D) {
    const int tid = threadIdx.x;

    // one block per row; grid is the number of rows
    const float4* row_x = reinterpret_cast<const float4*>(X + (size_t)blockIdx.x * D);
    float4* row_y = reinterpret_cast<float4*>(Y + (size_t)blockIdx.x * D);
    const int D4 = D >> 2;

    // per-thread partial reductions over the row
    float acc0 = REDUCE_INIT, acc1 = REDUCE_INIT;
    #pragma unroll 4
    for (int i = tid; i < D4; i += BLOCK_SIZE) {
        float4 x = row_x[i];
        acc0 = COMBINE(acc0, COMBINE(MAP_OP(x.x), MAP_OP(x.y)));
        acc1 = COMBINE(acc1, COMBINE(MAP_OP(x.z), MAP_OP(x.w)));
    }
    float acc = COMBINE(acc0, acc1);

    // butterfly all-reduce: every thread ends with the full row reduction
    __shared__ float merge_buf[BLOCK_SIZE];
    #pragma unroll
    for (int i = 1; i < BLOCK_SIZE; i *= 2) {
        merge_buf[tid] = acc;
        __syncthreads();
        acc = COMBINE(acc, merge_buf[(tid + i) % BLOCK_SIZE]);
        __syncthreads();
    }

    const float facc = FINALIZE(acc, D);
    #pragma unroll 4
    for (int i = tid; i < D4; i += BLOCK_SIZE) {
        float4 x = row_x[i];
        x.x = WRITE_OP(x.x, facc);
        x.y = WRITE_OP(x.y, facc);
        x.z = WRITE_OP(x.z, facc);
        x.w = WRITE_OP(x.w, facc);
        row_y[i] = x;
    }
}

// Launch one block per row (grid = row count, D = row length).
static inline void launch_reduction(const float* X, float* Y, int rows, int D) {
    reduction_kernel<<<rows, BLOCK_SIZE>>>(X, Y, D);
}

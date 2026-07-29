#pragma once
#include <cuda_runtime.h>

// Generic one-block-per-row reduction + elementwise-writeback, templated so each
// problem picks its hooks, block-reduction strategy, and BLOCK_SIZE at compile
// time (no macros). A row is reduced to a scalar, finalized, then every element
// is rewritten as a function of (element, finalized scalar).
//
// A solution instantiates one of the ReduceOps aliases below and launches it:
//   extern "C" void solution(const float* X, float* Y, size_t M, size_t N) {
//       launch_reduce<L2NormOps, 256>(X, Y, (int)M, (int)N);
//   }
//
// Row length D must be a multiple of 8 (the kernel processes two float4s per
// iteration, no scalar tail). BLOCK_SIZE must be a multiple of 32.

// ============================================================================
// Ops struct: bundles the five hooks a single-reduction kernel needs. Each hook
// is a static wrapper AND a raw function pointer so the TreeReduce
// template-template parameter can pick up Reduce and Identity.
// ============================================================================
template <float (*MapFn)(float),
          float (*FinalizeFn)(float, int),
          float (*OutputFn)(float, float),
          float (*ReduceFn)(float, float),
          float (*IdentityFn)()>
struct ReduceOps {
    __device__ __forceinline__ static float Map(float x)             { return MapFn(x); }
    __device__ __forceinline__ static float Finalize(float a, int D) { return FinalizeFn(a, D); }
    __device__ __forceinline__ static float Output(float x, float f) { return OutputFn(x, f); }
    __device__ __forceinline__ static float Reduce(float a, float b) { return ReduceFn(a, b); }
    __device__ __forceinline__ static float Identity()               { return IdentityFn(); }

    static constexpr float (*ReducePtr)(float, float) = ReduceFn;
    static constexpr float (*IdentityPtr)()           = IdentityFn;
};

// ============================================================================
// Two block-wide reduction strategies. Same signature so they're swappable as a
// template-template parameter to the kernel. Both leave every thread holding the
// full block reduction (pass 2 needs it in every thread).
// ============================================================================

template <int BLOCK_SIZE, float (*Reduce)(float, float), float (*Identity)()>
struct SmemTreeReduce {
    __device__ __forceinline__ static float apply(float acc) {
        const int tid = threadIdx.x;
        __shared__ float merge_buf[BLOCK_SIZE];

        #pragma unroll
        for (int i = 1; i < BLOCK_SIZE; i *= 2) {
            merge_buf[tid] = acc;
            __syncthreads();
            acc = Reduce(acc, merge_buf[(tid + i) % BLOCK_SIZE]);
            __syncthreads();
        }
        return acc;
    }
};

template <int BLOCK_SIZE, float (*Reduce)(float, float), float (*Identity)()>
struct WarpShuffleReduce {
    __device__ __forceinline__ static float apply(float acc) {
        constexpr int NUM_WARPS = BLOCK_SIZE / 32;
        const int tid = threadIdx.x;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float peer = __shfl_xor_sync(0xffffffff, acc, offset);
            acc = Reduce(acc, peer);
        }

        __shared__ float warp_vals[NUM_WARPS];
        const int lane    = tid & 31;
        const int warp_id = tid >> 5;
        if (lane == 0) warp_vals[warp_id] = acc;
        __syncthreads();

        __shared__ float block_val;
        if (warp_id == 0) {
            float v = (lane < NUM_WARPS) ? warp_vals[lane] : Identity();
            #pragma unroll
            for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
                float peer = __shfl_xor_sync(0xffffffff, v, offset);
                v = Reduce(v, peer);
            }
            if (lane == 0) block_val = v;
        }
        __syncthreads();
        return block_val;
    }
};

// ============================================================================
// The generic kernel: one block per row, float4-vectorized, TreeReduce strategy
// and BLOCK_SIZE chosen at compile time.
//
// VEC4 = number of float4s each thread loads/stores per loop iteration. VEC4=2
// (the default) is the dual-issue path: two 128-bit loads back to back so the
// compiler keeps two LDG.E.128 in flight to hide HBM latency, mirrored by two
// STG.E.128 in pass 2. VEC4=1 is the single-issue path. Two accumulators per
// float4 (2*VEC4 total) break the reduction dependency chain for ILP.
//
// Row length D must be a multiple of 4*VEC4 (8 for the default). BLOCK_SIZE must
// be a multiple of 32.
// ============================================================================
template <class Ops,
          int BLOCK_SIZE,
          int VEC4,
          template <int, float (*)(float,float), float (*)()> class TreeReduce>
__global__ void reduce_kernel(const float* __restrict__ X,
                              float* __restrict__ Y,
                              int D) {
    const int tid = threadIdx.x;
    const float4* row_x = reinterpret_cast<const float4*>(X + (size_t)blockIdx.x * D);
    float4*       row_y = reinterpret_cast<float4*>      (Y + (size_t)blockIdx.x * D);
    const int DV = D / (4 * VEC4);   // number of VEC4-groups (VEC4 float4s each) per row

    // Two accumulators per float4 break the reduction dependency chain.
    float acc[2 * VEC4];
    #pragma unroll
    for (int k = 0; k < 2 * VEC4; ++k) acc[k] = Ops::Identity();

    #pragma unroll 4
    for (int i = tid; i < DV; i += BLOCK_SIZE) {
        // VEC4 loads issued back to back (LDG.E.128 x VEC4), all in flight.
        #pragma unroll
        for (int v = 0; v < VEC4; ++v) {
            float4 x = row_x[i * VEC4 + v];
            acc[2 * v    ] = Ops::Reduce(acc[2 * v    ], Ops::Reduce(Ops::Map(x.x), Ops::Map(x.y)));
            acc[2 * v + 1] = Ops::Reduce(acc[2 * v + 1], Ops::Reduce(Ops::Map(x.z), Ops::Map(x.w)));
        }
    }
    float partial = acc[0];
    #pragma unroll
    for (int k = 1; k < 2 * VEC4; ++k) partial = Ops::Reduce(partial, acc[k]);

    const float f = Ops::Finalize(
        TreeReduce<BLOCK_SIZE, Ops::ReducePtr, Ops::IdentityPtr>::apply(partial),
        D);

    #pragma unroll 4
    for (int i = tid; i < DV; i += BLOCK_SIZE) {
        // VEC4 loads back to back, then VEC4 stores back to back.
        #pragma unroll
        for (int v = 0; v < VEC4; ++v) {
            float4 x = row_x[i * VEC4 + v];
            x.x = Ops::Output(x.x, f); x.y = Ops::Output(x.y, f);
            x.z = Ops::Output(x.z, f); x.w = Ops::Output(x.w, f);
            row_y[i * VEC4 + v] = x;
        }
    }
}

// ============================================================================
// Building blocks: Maps, Reduces, Identities, Finalizes, Outputs.
// ============================================================================

// Maps
__device__ __forceinline__ float map_identity(float x) { return x; }
__device__ __forceinline__ float map_square(float x)   { return x * x; }
__device__ __forceinline__ float map_abs(float x)      { return fabsf(x); }
__device__ __forceinline__ float map_exp(float x)      { return __expf(x); }

// Reduces
__device__ __forceinline__ float reduce_sum(float a, float b) { return a + b; }
__device__ __forceinline__ float reduce_max(float a, float b) { return fmaxf(a, b); }
__device__ __forceinline__ float reduce_min(float a, float b) { return fminf(a, b); }

// Identities
__device__ __forceinline__ float id_zero()    { return 0.0f; }
__device__ __forceinline__ float id_neg_inf() { return -INFINITY; }
__device__ __forceinline__ float id_pos_inf() { return  INFINITY; }

// Finalizes (all take D so they can scale by row size)
__device__ __forceinline__ float fin_identity(float a, int)  { return a; }
__device__ __forceinline__ float fin_recip(float a, int)     { return __fdividef(1.0f, a); }
__device__ __forceinline__ float fin_rsqrt(float a, int)     { return rsqrtf(a); }
__device__ __forceinline__ float fin_sqrt(float a, int)      { return sqrtf(a); }
__device__ __forceinline__ float fin_log(float a, int)       { return __logf(a); }
__device__ __forceinline__ float fin_mean(float a, int D)    { return __fdividef(a, (float)D); }
__device__ __forceinline__ float fin_rmsnorm(float a, int D) {
    return rsqrtf(__fdividef(a, (float)D) + 1e-5f);
}

// Outputs (x is the original element; f is the finalized scalar)
__device__ __forceinline__ float out_mul(float x, float f)    { return x * f; }
__device__ __forceinline__ float out_sub(float x, float f)    { return x - f; }
__device__ __forceinline__ float out_x  (float x, float)      { return x; }
__device__ __forceinline__ float out_expdiv(float x, float f) { return __fdividef(__expf(x), f); }

// ============================================================================
// Op catalog: one alias per Tensara reduction problem.
// ============================================================================

// L1 normalization:  y = x / sum(|x|)
using L1NormOps     = ReduceOps<map_abs,      fin_recip,    out_mul,    reduce_sum, id_zero>;

// L2 normalization:  y = x / sqrt(sum(x^2))
using L2NormOps     = ReduceOps<map_square,   fin_rsqrt,    out_mul,    reduce_sum, id_zero>;

// RMSNorm:           y = x * rsqrt(mean(x^2) + eps)
using RMSNormOps    = ReduceOps<map_square,   fin_rmsnorm,  out_mul,    reduce_sum, id_zero>;

// Mean subtract:     y = x - mean(x)
using MeanSubOps    = ReduceOps<map_identity, fin_mean,     out_sub,    reduce_sum, id_zero>;

// Max normalize (inf-norm): y = x / max(|x|)
using AbsMaxNormOps = ReduceOps<map_abs,      fin_recip,    out_mul,    reduce_max, id_zero>;

// Max normalize:     y = x / max(x)
using MaxNormOps    = ReduceOps<map_identity, fin_recip,    out_mul,    reduce_max, id_neg_inf>;

// Sum normalize:     y = x / sum(x)   (non-negative inputs)
using SumNormOps    = ReduceOps<map_identity, fin_recip,    out_mul,    reduce_sum, id_zero>;

// LogSoftmax (unstable): y = x - log(sum(exp(x)))   (bounded input only)
using LogSoftmaxOps = ReduceOps<map_exp,      fin_log,      out_sub,    reduce_sum, id_zero>;

// Softmax (unstable):    y = exp(x) / sum(exp(x))   (bounded input only)
using SoftmaxOps    = ReduceOps<map_exp,      fin_identity, out_expdiv, reduce_sum, id_zero>;

// ============================================================================
// Launcher: one block per row (grid = row count, D = row length).
//   VEC4       - float4s per iteration (2 = dual-issue default, 1 = single-issue)
//   TreeReduce - block reduction strategy (WarpShuffleReduce default, or SmemTreeReduce)
// e.g. launch_reduce<L2NormOps, 256>(...)                 // dual-issue, warp shuffle
//      launch_reduce<L2NormOps, 256, 1>(...)              // single-issue
//      launch_reduce<L2NormOps, 256, 2, SmemTreeReduce>(...)
// ============================================================================
template <class Ops, int BLOCK_SIZE, int VEC4 = 2,
          template <int, float (*)(float,float), float (*)()> class TreeReduce = WarpShuffleReduce>
static inline void launch_reduce(const float* X, float* Y, int rows, int D) {
    reduce_kernel<Ops, BLOCK_SIZE, VEC4, TreeReduce><<<rows, BLOCK_SIZE>>>(X, Y, D);
}

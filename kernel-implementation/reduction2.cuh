#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define NUM_WARPS (BLOCK_SIZE / 32)
#define tid threadIdx.x

// ============================================================================
// Ops struct: bundles the five hooks a single-reduction kernel needs.
// Each hook is stored as a static wrapper AND as a raw function pointer so the
// TreeReduce template-template parameter can pick up Reduce and Identity.
// ============================================================================
template <float (*MapFn)(float), float (*FinalizeFn)(float, int),
          float (*OutputFn)(float, float), float (*ReduceFn)(float, float),
          float (*IdentityFn)()>
struct ReduceOps {
  __device__ __forceinline__ static float Map(float x) { return MapFn(x); }
  __device__ __forceinline__ static float Finalize(float a, int D) {
    return FinalizeFn(a, D);
  }
  __device__ __forceinline__ static float Output(float x, float f) {
    return OutputFn(x, f);
  }
  __device__ __forceinline__ static float Reduce(float a, float b) {
    return ReduceFn(a, b);
  }
  __device__ __forceinline__ static float Identity() { return IdentityFn(); }

  static constexpr float (*ReducePtr)(float, float) = ReduceFn;
  static constexpr float (*IdentityPtr)() = IdentityFn;
};

// ============================================================================
// Two block-wide reduction strategies. Same signature so they're swappable
// as a template-template parameter to the kernel.
// ============================================================================

template <float (*Reduce)(float, float), float (*Identity)()>
struct SmemTreeReduce {
  __device__ __forceinline__ static float apply(float acc) {
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

template <float (*Reduce)(float, float), float (*Identity)()>
struct WarpShuffleReduce {
  __device__ __forceinline__ static float apply(float acc) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      float peer = __shfl_xor_sync(0xffffffff, acc, offset);
      acc = Reduce(acc, peer);
    }

    __shared__ float warp_vals[NUM_WARPS];
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0)
      warp_vals[warp_id] = acc;
    __syncthreads();

    __shared__ float block_val;
    if (warp_id == 0) {
      float v = (lane < NUM_WARPS) ? warp_vals[lane] : Identity();
#pragma unroll
      for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
        float peer = __shfl_xor_sync(0xffffffff, v, offset);
        v = Reduce(v, peer);
      }
      if (lane == 0)
        block_val = v;
    }
    __syncthreads();
    return block_val;
  }
};

// ============================================================================
// The generic kernel: one block per row, float4-vectorized, dual accumulators,
// TreeReduce strategy chosen at compile time.
// ============================================================================
template <class Ops,
          template <float (*)(float, float), float (*)()> class TreeReduce>
__global__ void reduce_kernel(const float *__restrict__ X,
                              float *__restrict__ Y, int D) {
  const float4 *row_x = reinterpret_cast<const float4 *>(X + blockIdx.x * D);
  float4 *row_y = reinterpret_cast<float4 *>(Y + blockIdx.x * D);
  const int D4 = D >> 2;

  float a0 = Ops::Identity();
  float a1 = Ops::Identity();

#pragma unroll 4
  for (int i = tid; i < D4; i += BLOCK_SIZE) {
    float4 x = row_x[i];
    a0 = Ops::Reduce(a0, Ops::Reduce(Ops::Map(x.x), Ops::Map(x.y)));
    a1 = Ops::Reduce(a1, Ops::Reduce(Ops::Map(x.z), Ops::Map(x.w)));
  }

  const float f = Ops::Finalize(
      TreeReduce<Ops::ReducePtr, Ops::IdentityPtr>::apply(Ops::Reduce(a0, a1)),
      D);

#pragma unroll 4
  for (int i = tid; i < D4; i += BLOCK_SIZE) {
    float4 x = row_x[i];
    x.x = Ops::Output(x.x, f);
    x.y = Ops::Output(x.y, f);
    x.z = Ops::Output(x.z, f);
    x.w = Ops::Output(x.w, f);
    row_y[i] = x;
  }
}

// ============================================================================
// Building blocks: Maps, Reduces, Identities, Finalizes, Outputs.
// ============================================================================

// Maps
__device__ __forceinline__ float map_identity(float x) { return x; }
__device__ __forceinline__ float map_square(float x) { return x * x; }
__device__ __forceinline__ float map_abs(float x) { return fabsf(x); }
__device__ __forceinline__ float map_exp(float x) { return __expf(x); }

// Reduces
__device__ __forceinline__ float reduce_sum(float a, float b) { return a + b; }
__device__ __forceinline__ float reduce_max(float a, float b) {
  return fmaxf(a, b);
}
__device__ __forceinline__ float reduce_min(float a, float b) {
  return fminf(a, b);
}

// Identities
__device__ __forceinline__ float id_zero() { return 0.0f; }
__device__ __forceinline__ float id_neg_inf() { return -INFINITY; }
__device__ __forceinline__ float id_pos_inf() { return INFINITY; }

// Finalizes (all take D so they can scale by row size)
__device__ __forceinline__ float fin_identity(float a, int) { return a; }
__device__ __forceinline__ float fin_recip(float a, int) {
  return __fdividef(1.0f, a);
}
__device__ __forceinline__ float fin_rsqrt(float a, int) { return rsqrtf(a); }
__device__ __forceinline__ float fin_sqrt(float a, int) { return sqrtf(a); }
__device__ __forceinline__ float fin_log(float a, int) { return __logf(a); }
__device__ __forceinline__ float fin_mean(float a, int D) {
  return __fdividef(a, (float)D);
}
__device__ __forceinline__ float fin_rmsnorm(float a, int D) {
  return rsqrtf(__fdividef(a, (float)D) + 1e-5f);
}

// Outputs
__device__ __forceinline__ float out_mul(float x, float f) { return x * f; }
__device__ __forceinline__ float out_sub(float x, float f) { return x - f; }
__device__ __forceinline__ float out_x(float x, float) { return x; }

// ============================================================================
// Op catalog: one alias per Tensara reduction problem.
// ============================================================================

// L1 normalization:      y = x / sum(|x|)
using L1NormOps = ReduceOps<map_abs, fin_recip, out_mul, reduce_sum, id_zero>;

// L2 normalization:      y = x / sqrt(sum(x^2))
using L2NormOps =
    ReduceOps<map_square, fin_rsqrt, out_mul, reduce_sum, id_zero>;

// RMSNorm:               y = x * rsqrt(mean(x^2) + eps)
using RMSNormOps =
    ReduceOps<map_square, fin_rmsnorm, out_mul, reduce_sum, id_zero>;

// Mean subtract:         y = x - mean(x)
using MeanSubOps =
    ReduceOps<map_identity, fin_mean, out_sub, reduce_sum, id_zero>;

// Max normalize:         y = x / max(x)
using MaxNormOps =
    ReduceOps<map_identity, fin_recip, out_mul, reduce_max, id_neg_inf>;

// Absmax normalize:      y = x / max(|x|)
using AbsMaxNormOps =
    ReduceOps<map_abs, fin_recip, out_mul, reduce_max, id_zero>;

// Sum normalize:         y = x / sum(x)   (for non-negative inputs)
using SumNormOps =
    ReduceOps<map_identity, fin_recip, out_mul, reduce_sum, id_zero>;

// LogSoftmax (unstable): y = x - log(sum(exp(x)))
// NOTE: overflows for x > ~88. Only correct when input range is bounded.
using LogSoftmaxOps = ReduceOps<map_exp, fin_log, out_sub, reduce_sum, id_zero>;

// ============================================================================
// Toggle: pick the reduction strategy and the op to expose as solution().
// ============================================================================

#define TREE_REDUCE WarpShuffleReduce // or SmemTreeReduce

#define OP_L1_NORM 1
#define OP_L2_NORM 2
#define OP_RMS_NORM 3
#define OP_MEAN_SUBTRACT 4
#define OP_MAX_NORM 5
#define OP_ABSMAX_NORM 6
#define OP_SUM_NORM 7
#define OP_LOG_SOFTMAX 8

#define OP OP_L1_NORM

template <class Ops>
static inline void launch_reduce(const float *X, float *Y, int M, int D) {
  reduce_kernel<Ops, TREE_REDUCE><<<M, BLOCK_SIZE>>>(X, Y, D);
}

extern "C" void solution(const float *input, float *output, size_t M,
                         size_t N) {
  int grid = static_cast<int>(M);
  int D = static_cast<int>(N);

#if OP == OP_L1_NORM
  launch_reduce<L1NormOps>(input, output, grid, D);
#elif OP == OP_L2_NORM
  launch_reduce<L2NormOps>(input, output, grid, D);
#elif OP == OP_RMS_NORM
  launch_reduce<RMSNormOps>(input, output, grid, D);
#elif OP == OP_MEAN_SUBTRACT
  launch_reduce<MeanSubOps>(input, output, grid, D);
#elif OP == OP_MAX_NORM
  launch_reduce<MaxNormOps>(input, output, grid, D);
#elif OP == OP_ABSMAX_NORM
  launch_reduce<AbsMaxNormOps>(input, output, grid, D);
#elif OP == OP_SUM_NORM
  launch_reduce<SumNormOps>(input, output, grid, D);
#elif OP == OP_LOG_SOFTMAX
  launch_reduce<LogSoftmaxOps>(input, output, grid, D);
#endif
}

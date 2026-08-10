#pragma once
// Reusable block-wise reduction -> single global atomic, for whole-tensor
// reductions to a scalar (losses, norms, dot products, ...).
//
// The block reduction itself is reused from reduction.cuh (SmemTreeReduce, the
// shared-memory all-to-all tree), so there's a single reduction implementation
// in the tree; this header just adds the "reduce then one atomicAdd per block"
// wrapper. BLOCK must be a power of two (SmemTreeReduce's rotation is mod
// BLOCK).

#include "reduction.cuh"
#include <cuda_runtime.h>

template <class LossImpl, float (*Reduce)(float, float)>
__device__ __forceinline__ float apply(float4 x, float4 y) {
  float4 r = {
      LossImpl::apply(x.x, y.x),
      LossImpl::apply(x.y, y.y),
      LossImpl::apply(x.z, y.z),
      LossImpl::apply(x.w, y.w),
  };
  float a = Reduce(r.x, r.y);
  float b = Reduce(r.z, r.w);
  return Reduce(a, b);
}

template <class LossImpl, float (*Reduce)(float, float), int BLOCK_SIZE,
          int VEC4 = 2>
__global__ void block_reduce(const float *predictions, const float *targets,
                             float *output, size_t n, float inv_n) {
  static_assert(VEC4 == 1 || VEC4 == 2,
                "Only support for 1 or 2 128-bit loads at a time");

  int base = (threadIdx.x + blockIdx.x * blockDim.x) * (VEC4 * 4);

  // Scalar tail for the last partial group of 8 so the tail matches the
  // vectorized path exactly.
  if (base + ((VEC4 * 4) - 1) >= n) {
    for (int i = base; i < n; ++i) {
      float r = LossImpl::apply(predictions[i], targets[i]);
      r *= inv_n;
      atomicAdd(output, r);
    }
  } else {

    float4 x0 = *reinterpret_cast<const float4 *>(predictions + base);
    float4 y0 = *reinterpret_cast<const float4 *>(targets + base);
    float r = apply<LossImpl, Reduce>(x0, y0);

    if (VEC4 == 2) {
      float4 x1 = *reinterpret_cast<const float4 *>(predictions + base + 4);
      float4 y1 = *reinterpret_cast<const float4 *>(targets + base + 4);
      r += apply<LossImpl, Reduce>(x1, y1);
    }

    r = SmemTreeReduce<BLOCK_SIZE, Reduce, id_zero>::apply(r);

    if (threadIdx.x != 0)
      return;

    r *= inv_n;
    atomicAdd(output, r);
  }
}

template <class LossImpl, float (*Reduce)(float, float), int BLOCK_SIZE,
          int VEC4 = 2>
void launch_block_reduce(const float *predictions, const float *targets,
                         float *output, size_t n) {
  int threads_needed = (n + ((VEC4 * 4) - 1)) / (VEC4 * 4);
  int grid = (threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;
  float inv_n = 1.0f / static_cast<float>(n);

  block_reduce<LossImpl, Reduce, BLOCK_SIZE, VEC4>
      <<<grid, BLOCK_SIZE>>>(predictions, targets, output, n, inv_n);
}

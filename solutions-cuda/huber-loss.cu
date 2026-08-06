// Solution stub for "huber-loss".
// The signature is derived from kernel-harnesses/huber-loss.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/huber-loss.exe
//   ./build/bin/huber-loss.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

#define BLOCK_SIZE 512

__global__ void _kernel(const float* predictions, const float* targets, float* output, size_t n, float inv_n) {

  int base = (threadIdx.x + blockIdx.x * blockDim.x) * 8;

  // Scalar tail for the last partial group of 8 so the tail matches the
  // vectorized path exactly.
  if (base + 7 >= n) {
    for (int i = base; i < n; ++i) {
      float x = predictions[i];
      float y = targets[i];
      float a = fabsf(x - y);

      float r = (a < 1) ? (0.5f * a * a) : (a - 0.5f);
      r *= inv_n;
      atomicAdd(output, r);
    }
  } else {

    float4 x0 = *reinterpret_cast<const float4*>(predictions + base);
    float4 x1 = *reinterpret_cast<const float4*>(predictions + base + 4);

    float4 y0 = *reinterpret_cast<const float4*>(targets + base);
    float4 y1 = *reinterpret_cast<const float4*>(targets + base + 4);

    float4 a0 = {
      fabsf(x0.x - y0.x),
      fabsf(x0.y - y0.y),
      fabsf(x0.z - y0.z),
      fabsf(x0.w - y0.w),
    };

    float4 a1= {
      fabsf(x1.x - y1.x),
      fabsf(x1.y - y1.y),
      fabsf(x1.z - y1.z),
      fabsf(x1.w - y1.w),
    };

    float4 r0 = {
      (a0.x < 1) ? (0.5f * a0.x * a0.x) : (a0.x - 0.5f),
      (a0.y < 1) ? (0.5f * a0.y * a0.y) : (a0.y - 0.5f),
      (a0.z < 1) ? (0.5f * a0.z * a0.z) : (a0.z - 0.5f),
      (a0.w < 1) ? (0.5f * a0.w * a0.w) : (a0.w - 0.5f),
    };

    float4 r1 = {
      (a1.x < 1) ? (0.5f * a1.x * a1.x) : (a1.x - 0.5f),
      (a1.y < 1) ? (0.5f * a1.y * a1.y) : (a1.y - 0.5f),
      (a1.z < 1) ? (0.5f * a1.z * a1.z) : (a1.z - 0.5f),
      (a1.w < 1) ? (0.5f * a1.w * a1.w) : (a1.w - 0.5f),
    };

    float acc0 = 
      r0.x +
      r0.y +
      r0.z +
      r0.w;
    float acc1 = 
      r1.x +
      r1.y +
      r1.z +
      r1.w;
    float r = acc0 + acc1;



#if 1
    const int tid = threadIdx.x;
    __shared__ float merge_buf[BLOCK_SIZE];

#pragma unroll
    for (int i = 1; i < BLOCK_SIZE; i *= 2) {
      merge_buf[tid] = r;
      __syncthreads();
      r += merge_buf[(tid + i) % BLOCK_SIZE];
      __syncthreads();
    }

    if (tid != 0)
      return;
#endif

    r *= inv_n;
    atomicAdd(output, r);
  }
}

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) {
  int threads_needed = (n + 7) / 8;
  int grid = (threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;
  float inv_n = 1.0f / static_cast<float>(n);

  _kernel<<<grid, BLOCK_SIZE>>>(predictions, targets, output, n, inv_n);
}

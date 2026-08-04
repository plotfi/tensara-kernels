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

#define BLOCK_SIZE 256

__global__ void _kernel(const float* predictions, const float* targets, float* output, size_t n) {
  unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) return;

  float x = predictions[i];
  float y = targets[i];
  float a = fabsf(x - y);

  float r = (a < 1) ? (0.5f * a * a) : (a - 0.5f);

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

  atomicAdd(output, r / (float)n);
}

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) {
  size_t threads_needed = n;
  const size_t grid = (threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;

  _kernel<<<grid, BLOCK_SIZE>>>(predictions, targets, output, n);
}

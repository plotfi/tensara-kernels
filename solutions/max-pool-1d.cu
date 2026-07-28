// Solution stub for "max-pool-1d".
// TODO: implement the body below. The signature is derived from
//       tensara-harnesses/max-pool-1d.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/max-pool-1d.exe
//   ./build/bin/max-pool-1d.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

__global__ void _kernel(const float* input, float* output, int k, int S, int P, int D, int H, int Hout) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= H) return;
    if (i >= Hout) return;

    float max = 0.0f;
    for (int m = 0; m < k; m++) {
        int RI = S * i + D * m - P;
        if (0 <= RI && RI < H) {
            max = fmaxf(max, input[RI]);
        }
    }

    output[i] = max;
}

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* input, int kernel_size, int stride, int padding, int dilation, float* output, size_t H) {
    const int BLOCK_SIZE = 256;
    int N = static_cast<int>(H);
    int Hout = (N + 2 * padding - kernel_size) / stride + 1;
    const int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    _kernel<<<grid, BLOCK_SIZE>>>(input, output, kernel_size, stride, padding, dilation, N, Hout);
}

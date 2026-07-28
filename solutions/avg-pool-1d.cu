// Solution for "avg-pool-1d".
// Signature matches tensara-harnesses/avg-pool-1d.cu.
//
// Build it (the build system auto-picks this file):
//   make build/bin/avg-pool-1d.exe
//   ./build/bin/avg-pool-1d.exe
#include <cuda_runtime.h>

__global__ void _kernel(const float* input, float* output, int k, int S, int P, int H, int Hout, float inv_k) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= H) return;
    if (i >= Hout) return;

    float sum = 0.0f;
    for (int m = 0; m < k; m++) {
        int RI = S * i + m - P;
        if (0 <= RI && RI < H) {
            sum += input[RI];
        }
    }

    output[i] = inv_k * sum;
}

// Note: input, output are device pointers
extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H) {
    const int BLOCK_SIZE = 256;
    int N = static_cast<int>(H);
    int Hout = (N + 2 * padding - kernel_size) / stride + 1;
    float inv_k = 1.0f / static_cast<float>(kernel_size);
    const int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    _kernel<<<grid, BLOCK_SIZE>>>(input, output, kernel_size, stride, padding, N, Hout, inv_k);
}

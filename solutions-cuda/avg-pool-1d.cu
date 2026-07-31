#include "../kernel-implementation/avg-pool-1d.cuh"

extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H) {
    const int BLOCK_SIZE = 256;
    int N = static_cast<int>(H);
    int Hout = (N + 2 * padding - kernel_size) / stride + 1;
    float inv_k = 1.0f / static_cast<float>(kernel_size);
    const int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    avgpool1d_kernel<<<grid, BLOCK_SIZE>>>(input, output, kernel_size, stride, padding, N, Hout, inv_k);
}

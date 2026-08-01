// Solution for "max-pool-1d": 1-D max pooling with dilation (padded taps ignored).
#include "../kernel-implementation/pooling.cuh"

extern "C" void solution(const float* input, int kernel_size, int stride, int padding, int dilation, float* output, size_t H) {
    const int BLOCK_SIZE = 256;
    int N = static_cast<int>(H);
    int Hout = pool_out_size(N, kernel_size, padding, dilation, stride);
    const int grid = (Hout + BLOCK_SIZE - 1) / BLOCK_SIZE;

    pool1d_kernel<MaxPoolOp><<<grid, BLOCK_SIZE>>>(input, output, kernel_size, stride, padding, dilation, N, Hout);
}

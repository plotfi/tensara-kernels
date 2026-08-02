// Solution for "conv1d-maxpool1d": fused same-length conv1d followed by max
// pooling, with the conv output staged in shared memory (never in global).
#include "../kernel-implementation/conv1d-maxpool1d.cuh"

extern "C" void solution(const float* input, const float* weight, float* output,
                         size_t N, size_t K,
                         int kernel_size, int stride, int padding, int dilation) {
    if (N == 0) return;
    launch_conv1d_maxpool1d(input, weight, output,
                            static_cast<int>(N), static_cast<int>(K),
                            kernel_size, stride, padding, dilation);
}

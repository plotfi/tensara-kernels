#include "../kernel-implementation/grayscale.cuh"

extern "C" void solution(const float* rgb_image, float* grayscale_output, size_t height, size_t width, size_t channels) {
    int N = static_cast<int>(height * width);
    if (N == 0) return;

    int threads_needed = (N + 3) / 4;
    const int grid = (threads_needed + GRAYSCALE_BLOCK_SIZE - 1) / GRAYSCALE_BLOCK_SIZE;
    grayscale_kernel<<<grid, GRAYSCALE_BLOCK_SIZE>>>(rgb_image, grayscale_output, N);
}

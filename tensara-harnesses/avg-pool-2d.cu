#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H, size_t W);

int main() {
    harness::begin("avg-pool-2d");

    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
    size_t H = 32;
    size_t W = 32;

    harness::Buffer<float> input(H * W);
    harness::Buffer<float> output(H * W);

    input.fill_random();

    BENCHMARK(solution(input, kernel_size, stride, padding, output, H, W));

    output.preview("output");

    printf("Done.\n");
    return 0;
}

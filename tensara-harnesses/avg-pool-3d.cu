#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H, size_t W, size_t D);

int main() {
    harness::begin("avg-pool-3d");

    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
    size_t H = 16;
    size_t W = 16;
    size_t D = 16;

    harness::Buffer<float> input(H * W * D);
    harness::Buffer<float> output(H * W * D);

    input.fill_random();

    BENCHMARK(solution(input, kernel_size, stride, padding, output, H, W, D));

    output.preview("output");

    harness::end();
}

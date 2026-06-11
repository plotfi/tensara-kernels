#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input, int kernel_size, int stride, int padding, int dilation, float* output, size_t H);

int main() {
    harness::begin("max-pool-1d");

    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
    int dilation = 1;
    size_t H = 1024;

    harness::Buffer<float> input(H);
    harness::Buffer<float> output(H);

    input.fill_random();

    BENCHMARK(solution(input, kernel_size, stride, padding, dilation, output, H));

    output.preview("output");

    harness::end();
}

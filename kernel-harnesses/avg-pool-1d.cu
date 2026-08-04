#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H);

int main() {
    tensor::begin("avg-pool-1d");

    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
    size_t H = tensor::bench_size("H", 1024);

    tensor::Buffer<float> input(H);
    tensor::Buffer<float> output(H);

    input.fill_random();

    BENCHMARK(solution(input, kernel_size, stride, padding, output, H));

    output.preview("output");

    tensor::end();
}

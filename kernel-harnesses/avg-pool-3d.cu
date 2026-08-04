#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H, size_t W, size_t D);

int main() {
    tensor::begin("avg-pool-3d");

    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
    size_t H = tensor::bench_size("H", 16);
    size_t W = tensor::bench_size("W", 16);
    size_t D = tensor::bench_size("D", 16);

    tensor::Buffer<float> input(H * W * D);
    tensor::Buffer<float> output(H * W * D);

    input.fill_random();

    BENCHMARK(solution(input, kernel_size, stride, padding, output, H, W, D));

    output.preview("output");

    tensor::end();
}

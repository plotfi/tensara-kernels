#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input, size_t W, float* output, size_t N);

int main() {
    tensor::begin("running-sum-1d");

    size_t W = 5;
    size_t N = tensor::bench_size("N", 1024);

    tensor::Buffer<float> input(N);
    tensor::Buffer<float> output(N);

    input.fill_random();

    BENCHMARK(solution(input, W, output, N));

    output.preview("output");

    tensor::end();
}

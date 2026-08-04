#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input, float alpha, float* output, size_t n, size_t m);

int main() {
    tensor::begin("leaky-relu");

    float alpha = 0.01f;
    size_t n = tensor::bench_size("N", 64);
    size_t m = tensor::bench_size("M", 64);

    tensor::Buffer<float> input(n * m);
    tensor::Buffer<float> output(n * m);

    input.fill_random();

    BENCHMARK(solution(input, alpha, output, n, m));

    output.preview("output");

    tensor::end();
}

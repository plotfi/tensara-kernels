#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t n, size_t k);

int main() {
    tensor::begin("matrix-multiplication");

    size_t m = tensor::bench_size("M", 64);
    size_t n = tensor::bench_size("N", 64);
    size_t k = tensor::bench_size("K", 64);

    tensor::Buffer<float> input_a(m * k);
    tensor::Buffer<float> input_b(k * n);
    tensor::Buffer<float> output_c(m * n);

    input_a.fill_random();
    input_b.fill_random();

    BENCHMARK(solution(input_a, input_b, output_c, m, n, k));

    output_c.preview("output_c");

    tensor::end();
}

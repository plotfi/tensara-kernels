#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t n);

int main() {
    tensor::begin("square-matmul");

    size_t n = 64;

    tensor::Buffer<float> input_a(n * n);
    tensor::Buffer<float> input_b(n * n);
    tensor::Buffer<float> output_c(n * n);

    input_a.fill_random();
    input_b.fill_random();

    BENCHMARK(solution(input_a, input_b, output_c, n));

    output_c.preview("output_c");

    tensor::end();
}

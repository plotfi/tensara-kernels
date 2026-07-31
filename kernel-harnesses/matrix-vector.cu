#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t k);

int main() {
    tensor::begin("matrix-vector");

    size_t m = 64;
    size_t k = 64;

    tensor::Buffer<float> input_a(m * k);
    tensor::Buffer<float> input_b(k);
    tensor::Buffer<float> output_c(m);

    input_a.fill_random();
    input_b.fill_random();

    BENCHMARK(solution(input_a, input_b, output_c, m, k));

    output_c.preview("output_c");

    tensor::end();
}

#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t n, size_t k);

int main() {
    harness::begin("matrix-multiplication");

    size_t m = 64;
    size_t n = 64;
    size_t k = 64;

    harness::Buffer<float> input_a(m * k);
    harness::Buffer<float> input_b(k * n);
    harness::Buffer<float> output_c(m * n);

    input_a.fill_random();
    input_b.fill_random();

    BENCHMARK(solution(input_a, input_b, output_c, m, n, k));

    output_c.preview("output_c");

    harness::end();
}

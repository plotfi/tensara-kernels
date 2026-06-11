#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t n);

int main() {
    harness::begin("symmetric-matmul");

    size_t n = 64;

    harness::Buffer<float> input_a(n * n);
    harness::Buffer<float> input_b(n * n);
    harness::Buffer<float> output_c(n * n);

    input_a.fill_random();
    input_b.fill_random();

    BENCHMARK(solution(input_a, input_b, output_c, n));

    output_c.preview("output_c");

    harness::end();
}

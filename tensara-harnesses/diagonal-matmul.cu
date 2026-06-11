#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* diagonal_a, const float* input_b, float* output_c, size_t n, size_t m);

int main() {
    harness::begin("diagonal-matmul");

    size_t n = 64;
    size_t m = 64;

    harness::Buffer<float> diagonal_a(n);
    harness::Buffer<float> input_b(n * m);
    harness::Buffer<float> output_c(n * m);

    diagonal_a.fill_random();
    input_b.fill_random();

    BENCHMARK(solution(diagonal_a, input_b, output_c, n, m));

    output_c.preview("output_c");

    printf("Done.\n");
    return 0;
}

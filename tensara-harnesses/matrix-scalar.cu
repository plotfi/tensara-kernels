#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input_matrix, float scalar, float* output_matrix, size_t n);

int main() {
    harness::begin("matrix-scalar");

    float scalar = 2.5f;
    size_t n = 64;

    harness::Buffer<float> input_matrix(n * n);
    harness::Buffer<float> output_matrix(n * n);

    input_matrix.fill_random();

    BENCHMARK(solution(input_matrix, scalar, output_matrix, n));

    output_matrix.preview("output_matrix");

    printf("Done.\n");
    return 0;
}

#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input_matrix, size_t n, float* output_matrix, size_t size);

int main() {
    harness::begin("matrix-power");

    size_t n = 3;
    size_t size = 8;

    harness::Buffer<float> input_matrix(size * size);
    harness::Buffer<float> output_matrix(size * size);

    input_matrix.fill_random();

    BENCHMARK(solution(input_matrix, n, output_matrix, size));

    output_matrix.preview("output_matrix");

    printf("Done.\n");
    return 0;
}

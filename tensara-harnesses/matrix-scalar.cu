#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input_matrix, float scalar, float* output_matrix, size_t n);

int main() {
    tensor::begin("matrix-scalar");

    float scalar = 2.5f;
    size_t n = 64;

    tensor::Buffer<float> input_matrix(n * n);
    tensor::Buffer<float> output_matrix(n * n);

    input_matrix.fill_random();

    BENCHMARK(solution(input_matrix, scalar, output_matrix, n));

    output_matrix.preview("output_matrix");

    tensor::end();
}

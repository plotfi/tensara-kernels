#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input_matrix, size_t n, float* output_matrix, size_t size);

int main() {
    tensor::begin("matrix-power");

    size_t n = tensor::bench_size("N", 3);
    size_t size = tensor::bench_size("SIZE", 8);

    tensor::Buffer<float> input_matrix(size * size);
    tensor::Buffer<float> output_matrix(size * size);

    input_matrix.fill_random();

    BENCHMARK(solution(input_matrix, n, output_matrix, size));

    output_matrix.preview("output_matrix");

    tensor::end();
}

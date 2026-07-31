#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* adj_matrix, float* output, size_t n);

int main() {
    tensor::begin("all-pairs-shortest-path");

    size_t n = 64;

    tensor::Buffer<float> adj_matrix(n * n);
    tensor::Buffer<float> output(n * n);

    adj_matrix.fill_random();

    BENCHMARK(solution(adj_matrix, output, n));

    output.preview("output");

    tensor::end();
}

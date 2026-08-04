#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* d_adj_matrix, int source, float* d_distances, size_t n);

int main() {
    tensor::begin("shortest-path");

    int source = 0;
    size_t n = tensor::bench_size("N", 64);

    tensor::Buffer<float> d_adj_matrix(n * n);
    tensor::Buffer<float> d_distances(n);

    d_adj_matrix.fill_random();

    BENCHMARK(solution(d_adj_matrix, source, d_distances, n));

    d_distances.preview("d_distances");

    tensor::end();
}

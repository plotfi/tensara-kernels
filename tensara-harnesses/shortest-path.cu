#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* d_adj_matrix, int source, float* d_distances, size_t n);

int main() {
    harness::begin("shortest-path");

    int source = 0;
    size_t n = 64;

    harness::Buffer<float> d_adj_matrix(n * n);
    harness::Buffer<float> d_distances(n);

    d_adj_matrix.fill_random();

    BENCHMARK(solution(d_adj_matrix, source, d_distances, n));

    d_distances.preview("d_distances");

    printf("Done.\n");
    return 0;
}

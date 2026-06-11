#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* adj_matrix, float* output, size_t n);

int main() {
    harness::begin("all-pairs-shortest-path");

    size_t n = 64;

    harness::Buffer<float> adj_matrix(n * n);
    harness::Buffer<float> output(n * n);

    adj_matrix.fill_random();

    BENCHMARK(solution(adj_matrix, output, n));

    output.preview("output");

    printf("Done.\n");
    return 0;
}

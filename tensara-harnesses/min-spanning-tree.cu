#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* A, float* min_weight, size_t n);

int main() {
    harness::begin("min-spanning-tree");

    size_t n = 64;

    harness::Buffer<float> A(n * n);
    harness::Buffer<float> min_weight(1);

    A.fill_random();

    BENCHMARK(solution(A, min_weight, n));

    min_weight.preview("min_weight");

    printf("Done.\n");
    return 0;
}

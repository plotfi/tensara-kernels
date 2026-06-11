#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const int* a, int* b, size_t n);

int main() {
    harness::begin("array-sort");

    size_t n = 1024;

    harness::Buffer<int> a(n);
    harness::Buffer<int> b(n);

    a.fill_random();

    BENCHMARK(solution(a, b, n));

    b.preview("b");

    printf("Done.\n");
    return 0;
}

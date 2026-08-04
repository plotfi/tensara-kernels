#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const int* a, int* b, size_t n);

int main() {
    tensor::begin("array-sort");

    size_t n = tensor::bench_size("N", 1024);

    tensor::Buffer<int> a(n);
    tensor::Buffer<int> b(n);

    a.fill_random();

    BENCHMARK(solution(a, b, n));

    b.preview("b");

    tensor::end();
}

#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* A, const float* B, float* C, size_t n, size_t m, size_t k, size_t l);

int main() {
    tensor::begin("matmul-3d");

    size_t n = tensor::bench_size("N", 4);
    size_t m = tensor::bench_size("M", 64);
    size_t k = tensor::bench_size("K", 64);
    size_t l = tensor::bench_size("L", 32);

    tensor::Buffer<float> A(n * m * k);
    tensor::Buffer<float> B(k * l);
    tensor::Buffer<float> C(n * m * l);

    A.fill_random();
    B.fill_random();

    BENCHMARK(solution(A, B, C, n, m, k, l));

    C.preview("C");

    tensor::end();
}

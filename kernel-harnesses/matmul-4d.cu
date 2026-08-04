#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* A, const float* B, float* C, size_t b, size_t i, size_t j, size_t l, size_t k);

int main() {
    tensor::begin("matmul-4d");

    size_t b = tensor::bench_size("B", 2);
    size_t i = tensor::bench_size("I", 4);
    size_t j = tensor::bench_size("J", 32);
    size_t l = tensor::bench_size("L", 32);
    size_t k = tensor::bench_size("K", 16);

    tensor::Buffer<float> A(b * i * j * k);
    tensor::Buffer<float> B(k * l);
    tensor::Buffer<float> C(b * i * j * l);

    A.fill_random();
    B.fill_random();

    BENCHMARK(solution(A, B, C, b, i, j, l, k));

    C.preview("C");

    tensor::end();
}

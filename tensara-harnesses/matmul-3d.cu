#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* A, const float* B, float* C, size_t n, size_t m, size_t k, size_t l);

int main() {
    tensor::begin("matmul-3d");

    size_t n = 4;
    size_t m = 64;
    size_t k = 64;
    size_t l = 32;

    tensor::Buffer<float> A(n * m * k);
    tensor::Buffer<float> B(k * l);
    tensor::Buffer<float> C(n * m * l);

    A.fill_random();
    B.fill_random();

    BENCHMARK(solution(A, B, C, n, m, k, l));

    C.preview("C");

    tensor::end();
}

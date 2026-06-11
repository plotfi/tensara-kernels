#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* A, const float* B, float* C, size_t n, size_t m, size_t k, size_t l);

int main() {
    harness::begin("matmul-3d");

    size_t n = 4;
    size_t m = 64;
    size_t k = 64;
    size_t l = 32;

    harness::Buffer<float> A(n * m * k);
    harness::Buffer<float> B(k * l);
    harness::Buffer<float> C(n * m * l);

    A.fill_random();
    B.fill_random();

    BENCHMARK(solution(A, B, C, n, m, k, l));

    C.preview("C");

    harness::end();
}

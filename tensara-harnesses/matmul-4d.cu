#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* A, const float* B, float* C, size_t b, size_t i, size_t j, size_t l, size_t k);

int main() {
    harness::begin("matmul-4d");

    size_t b = 2;
    size_t i = 4;
    size_t j = 32;
    size_t l = 32;
    size_t k = 16;

    harness::Buffer<float> A(b * i * j * k);
    harness::Buffer<float> B(k * l);
    harness::Buffer<float> C(b * i * j * l);

    A.fill_random();
    B.fill_random();

    BENCHMARK(solution(A, B, C, b, i, j, l, k));

    C.preview("C");

    harness::end();
}

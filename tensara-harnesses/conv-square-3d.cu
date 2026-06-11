#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* A, const float* B, float* C, size_t size, size_t K);

int main() {
    harness::begin("conv-square-3d");

    size_t size = 16;
    size_t K = 3;

    harness::Buffer<float> A(size * size * size);
    harness::Buffer<float> B(K * K * K);
    harness::Buffer<float> C(size * size * size);

    A.fill_random();
    B.fill_random();

    BENCHMARK(solution(A, B, C, size, K));

    C.preview("C");

    harness::end();
}

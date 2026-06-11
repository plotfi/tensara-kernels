#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* A, const float* W, const float* b, float* C, size_t B, size_t N, size_t M);

int main() {
    harness::begin("gemm-relu");

    size_t B = 8;
    size_t N = 64;
    size_t M = 32;

    harness::Buffer<float> A(B * N);
    harness::Buffer<float> W(M * N);
    harness::Buffer<float> b(M);
    harness::Buffer<float> C(B * M);

    A.fill_random();
    W.fill_random();
    b.fill_random();

    BENCHMARK(solution(A, W, b, C, B, N, M));

    C.preview("C");

    printf("Done.\n");
    return 0;
}

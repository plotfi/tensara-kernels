#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* A, const float* B, float* output, size_t M, size_t N, size_t K);

int main() {
    harness::begin("matmul-sigmoid-sum");

    size_t M = 64;
    size_t N = 64;
    size_t K = 64;

    harness::Buffer<float> A(M * K);
    harness::Buffer<float> B(K * N);
    harness::Buffer<float> output(1);

    A.fill_random();
    B.fill_random();

    BENCHMARK(solution(A, B, output, M, N, K));

    output.preview("output");

    printf("Done.\n");
    return 0;
}

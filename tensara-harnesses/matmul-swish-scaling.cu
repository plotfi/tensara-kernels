#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* A, const float* B, float scale, float* output, size_t M, size_t N, size_t K);

int main() {
    harness::begin("matmul-swish-scaling");

    float scale = 1.0f;
    size_t M = 64;
    size_t N = 64;
    size_t K = 64;

    harness::Buffer<float> A(M * K);
    harness::Buffer<float> B(K * N);
    harness::Buffer<float> output(M * N);

    A.fill_random();
    B.fill_random();

    BENCHMARK(solution(A, B, scale, output, M, N, K));

    output.preview("output");

    harness::end();
}

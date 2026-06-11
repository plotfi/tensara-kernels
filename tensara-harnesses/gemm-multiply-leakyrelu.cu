#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* A, const float* B, const float* C, float alpha, float* output, size_t M, size_t N, size_t K);

int main() {
    harness::begin("gemm-multiply-leakyrelu");

    float alpha = 0.01f;
    size_t M = 64;
    size_t N = 64;
    size_t K = 64;

    harness::Buffer<float> A(M * K);
    harness::Buffer<float> B(K * N);
    harness::Buffer<float> C(M * N);
    harness::Buffer<float> output(M * N);

    A.fill_random();
    B.fill_random();
    C.fill_random();

    BENCHMARK(solution(A, B, C, alpha, output, M, N, K));

    output.preview("output");

    harness::end();
}

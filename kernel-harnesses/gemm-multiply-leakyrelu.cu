#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* A, const float* B, const float* C, float alpha, float* output, size_t M, size_t N, size_t K);

int main() {
    tensor::begin("gemm-multiply-leakyrelu");

    float alpha = 0.01f;
    size_t M = 64;
    size_t N = 64;
    size_t K = 64;

    tensor::Buffer<float> A(M * K);
    tensor::Buffer<float> B(K * N);
    tensor::Buffer<float> C(M * N);
    tensor::Buffer<float> output(M * N);

    A.fill_random();
    B.fill_random();
    C.fill_random();

    BENCHMARK(solution(A, B, C, alpha, output, M, N, K));

    output.preview("output");

    tensor::end();
}

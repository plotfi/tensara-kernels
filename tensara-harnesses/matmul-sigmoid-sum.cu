#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* A, const float* B, float* output, size_t M, size_t N, size_t K);

int main() {
    tensor::begin("matmul-sigmoid-sum");

    size_t M = 64;
    size_t N = 64;
    size_t K = 64;

    tensor::Buffer<float> A(M * K);
    tensor::Buffer<float> B(K * N);
    tensor::Buffer<float> output(1);

    A.fill_random();
    B.fill_random();

    BENCHMARK(solution(A, B, output, M, N, K));

    output.preview("output");

    tensor::end();
}

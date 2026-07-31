#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K);

int main() {
    tensor::begin("conv-1d");

    size_t N = 1024;
    size_t K = 5;

    tensor::Buffer<float> A(N);
    tensor::Buffer<float> B(K);
    tensor::Buffer<float> C(N);

    A.fill_random();
    B.fill_random();

    BENCHMARK(solution(A, B, C, N, K));

    C.preview("C");

    tensor::end();
}

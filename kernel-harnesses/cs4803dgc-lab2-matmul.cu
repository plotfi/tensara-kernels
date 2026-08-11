#include "../tensor-lib/tensor.cuh"

// CS4803 Lab 2 — tiled/shared-memory matrix multiply, C = A * B (square n x n).
extern "C" void solution(const float* A, const float* B, float* C, size_t n);

int main() {
    tensor::begin("cs4803dgc-lab2-matmul");

    size_t n = 512;

    tensor::Buffer<float> A(n * n);
    tensor::Buffer<float> B(n * n);
    tensor::Buffer<float> C(n * n);

    A.fill_random();
    B.fill_random();

    BENCHMARK(solution(A, B, C, n));

    C.preview("C");

    tensor::end();
}

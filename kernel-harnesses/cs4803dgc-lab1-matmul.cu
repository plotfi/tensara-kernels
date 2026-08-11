#include "../tensor-lib/tensor.cuh"

// CS4803 Lab 1 — naive single-block matrix multiply, C = A * B (square n x n).
// The kernel is single-block (threadIdx only), so n is capped at 32 (<=1024
// threads). Not env-scalable — this is the naive reference version.
extern "C" void solution(const float* A, const float* B, float* C, size_t n);

int main() {
    tensor::begin("cs4803dgc-lab1-matmul");

    size_t n = 32;   // single block: n*n <= 1024 threads

    tensor::Buffer<float> A(n * n);
    tensor::Buffer<float> B(n * n);
    tensor::Buffer<float> C(n * n);

    A.fill_random();
    B.fill_random();

    BENCHMARK(solution(A, B, C, n));

    C.preview("C");

    tensor::end();
}

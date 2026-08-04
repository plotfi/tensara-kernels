#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* A, const float* W, const float* b, float* C, size_t B, size_t N, size_t M);

int main() {
    tensor::begin("gemm-relu");

    size_t B = tensor::bench_size("B", 8);
    size_t N = tensor::bench_size("N", 64);
    size_t M = tensor::bench_size("M", 32);

    tensor::Buffer<float> A(B * N);
    tensor::Buffer<float> W(M * N);
    tensor::Buffer<float> b(M);
    tensor::Buffer<float> C(B * M);

    A.fill_random();
    W.fill_random();
    b.fill_random();

    BENCHMARK(solution(A, W, b, C, B, N, M));

    C.preview("C");

    tensor::end();
}

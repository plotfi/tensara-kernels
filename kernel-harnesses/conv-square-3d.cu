#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* A, const float* B, float* C, size_t size, size_t K);

int main() {
    tensor::begin("conv-square-3d");

    size_t size = tensor::bench_size("SIZE", 16);
    size_t K = 3;

    tensor::Buffer<float> A(size * size * size);
    tensor::Buffer<float> B(K * K * K);
    tensor::Buffer<float> C(size * size * size);

    A.fill_random();
    B.fill_random();

    BENCHMARK(solution(A, B, C, size, K));

    C.preview("C");

    tensor::end();
}

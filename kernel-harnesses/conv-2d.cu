#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* A, const float* B, float* C, size_t H, size_t W, size_t Kh, size_t Kw);

int main() {
    tensor::begin("conv-2d");

    size_t H = tensor::bench_size("H", 64);
    size_t W = tensor::bench_size("W", 64);
    size_t Kh = 3;
    size_t Kw = 3;

    tensor::Buffer<float> A(H * W);
    tensor::Buffer<float> B(Kh * Kw);
    tensor::Buffer<float> C(H * W);

    A.fill_random();
    B.fill_random();

    BENCHMARK(solution(A, B, C, H, W, Kh, Kw));

    C.preview("C");

    tensor::end();
}

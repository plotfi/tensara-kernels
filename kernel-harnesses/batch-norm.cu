#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* X, float* Y, size_t B, size_t F, size_t D1, size_t D2);

int main() {
    tensor::begin("batch-norm");

    size_t B = tensor::bench_size("B", 2);
    size_t F = tensor::bench_size("F", 4);
    size_t D1 = tensor::bench_size("D1", 8);
    size_t D2 = tensor::bench_size("D2", 8);

    tensor::Buffer<float> X(B * F * D1 * D2);
    tensor::Buffer<float> Y(B * F * D1 * D2);

    X.fill_random();

    BENCHMARK(solution(X, Y, B, F, D1, D2));

    Y.preview("Y");

    tensor::end();
}

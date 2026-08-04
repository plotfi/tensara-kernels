#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* X, float* Y, size_t B, size_t D);

int main() {
    tensor::begin("mean-subtract");

    size_t B = tensor::bench_size("B", 8);
    size_t D = tensor::bench_size("D", 64);

    tensor::Buffer<float> X(B * D);
    tensor::Buffer<float> Y(B * D);

    X.fill_random();

    BENCHMARK(solution(X, Y, B, D));

    Y.preview("Y");

    tensor::end();
}

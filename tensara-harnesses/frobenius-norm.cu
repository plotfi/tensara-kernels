#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* X, float* Y, size_t size);

int main() {
    tensor::begin("frobenius-norm");

    size_t size = 4096;

    tensor::Buffer<float> X(size);
    tensor::Buffer<float> Y(size);

    X.fill_random();

    BENCHMARK(solution(X, Y, size));

    Y.preview("Y");

    tensor::end();
}

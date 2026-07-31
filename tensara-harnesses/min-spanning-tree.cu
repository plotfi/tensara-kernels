#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* A, float* min_weight, size_t n);

int main() {
    tensor::begin("min-spanning-tree");

    size_t n = 64;

    tensor::Buffer<float> A(n * n);
    tensor::Buffer<float> min_weight(1);

    A.fill_random();

    BENCHMARK(solution(A, min_weight, n));

    min_weight.preview("min_weight");

    tensor::end();
}

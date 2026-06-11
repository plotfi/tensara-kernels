#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* X, float* Y, size_t size);

int main() {
    harness::begin("frobenius-norm");

    size_t size = 4096;

    harness::Buffer<float> X(size);
    harness::Buffer<float> Y(size);

    X.fill_random();

    BENCHMARK(solution(X, Y, size));

    Y.preview("Y");

    harness::end();
}

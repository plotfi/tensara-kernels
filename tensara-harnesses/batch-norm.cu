#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* X, float* Y, size_t B, size_t F, size_t D1, size_t D2);

int main() {
    harness::begin("batch-norm");

    size_t B = 2;
    size_t F = 4;
    size_t D1 = 8;
    size_t D2 = 8;

    harness::Buffer<float> X(B * F * D1 * D2);
    harness::Buffer<float> Y(B * F * D1 * D2);

    X.fill_random();

    BENCHMARK(solution(X, Y, B, F, D1, D2));

    Y.preview("Y");

    harness::end();
}

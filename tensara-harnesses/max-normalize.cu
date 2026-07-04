#include "../kernel-implementation/harness.cuh"
#include "../kernel-implementation/max-normalize.cu"

extern "C" void solution(const float* X, float* Y, size_t B, size_t D);

int main() {
    harness::begin("max-normalize");

    size_t B = 8;
    size_t D = 64;

    harness::Buffer<float> X(B * D);
    harness::Buffer<float> Y(B * D);

    X.fill_random();

    BENCHMARK(solution(X, Y, B, D));

    Y.preview("Y");

    harness::end();
}

#include "../kernel-implementation/harness.cuh"
#include "../kernel-implementation/rms-norm.cu"

extern "C" void solution(const float* X, float* Y, size_t B, size_t N);

int main() {
    harness::begin("rms-norm");

    size_t B = 8;
    size_t N = 64;

    harness::Buffer<float> X(B * N);
    harness::Buffer<float> Y(B * N);

    X.fill_random();

    BENCHMARK(solution(X, Y, B, N));

    Y.preview("Y");

    harness::end();
}

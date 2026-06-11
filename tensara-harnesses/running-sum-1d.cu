#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input, size_t W, float* output, size_t N);

int main() {
    harness::begin("running-sum-1d");

    size_t W = 5;
    size_t N = 1024;

    harness::Buffer<float> input(N);
    harness::Buffer<float> output(N);

    input.fill_random();

    BENCHMARK(solution(input, W, output, N));

    output.preview("output");

    harness::end();
}

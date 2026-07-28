#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input, float* output, size_t M, size_t N);

int main() {
    harness::begin("log-softmax");

    size_t M = 64;
    size_t N = 64;

    harness::Buffer<float> input(M * N);
    harness::Buffer<float> output(M * N);

    input.fill_random();

    BENCHMARK(solution(input, output, M, N));

    output.preview("output");

    harness::end();
}

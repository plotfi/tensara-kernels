#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input, float* output, size_t N);

int main() {
    harness::begin("cumprod");

    size_t N = 1024;

    harness::Buffer<float> input(N);
    harness::Buffer<float> output(N);

    input.fill_random();

    BENCHMARK(solution(input, output, N));

    output.preview("output");

    printf("Done.\n");
    return 0;
}

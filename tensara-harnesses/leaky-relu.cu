#include "../kernel-implementation/harness.cuh"
#include "../kernel-implementation/activations.cu"

extern "C" void solution(const float* input, float alpha, float* output, size_t n, size_t m);

int main() {
    harness::begin("leaky-relu");

    float alpha = 0.01f;
    size_t n = 64;
    size_t m = 64;

    harness::Buffer<float> input(n * m);
    harness::Buffer<float> output(n * m);

    input.fill_random();

    BENCHMARK(solution(input, alpha, output, n, m));

    output.preview("output");

    printf("Done.\n");
    return 0;
}

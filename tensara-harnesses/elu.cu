#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input, float* output, size_t n, size_t m, float alpha);

int main() {
    harness::begin("elu");

    size_t n = 64;
    size_t m = 64;
    float alpha = 1.0f;

    harness::Buffer<float> input(n * m);
    harness::Buffer<float> output(n * m);

    input.fill_random();

    BENCHMARK(solution(input, output, n, m, alpha));

    output.preview("output");

    harness::end();
}

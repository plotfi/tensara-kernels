#include "../kernel-implementation/harness.cuh"
#include "../kernel-implementation/activations.cu"

extern "C" void solution(const float* input, float* output, size_t n, size_t m);

int main() {
    harness::begin("hard-sigmoid");

    size_t n = 64;
    size_t m = 64;

    harness::Buffer<float> input(n * m);
    harness::Buffer<float> output(n * m);

    input.fill_random();

    BENCHMARK(solution(input, output, n, m));

    output.preview("output");

    harness::end();
}

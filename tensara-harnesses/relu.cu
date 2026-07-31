#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input, float* output, size_t n, size_t m);

int main() {
    tensor::begin("relu");

    size_t n = 64;
    size_t m = 64;

    tensor::Buffer<float> input(n * m);
    tensor::Buffer<float> output(n * m);

    input.fill_random();

    BENCHMARK(solution(input, output, n, m));

    output.preview("output");

    tensor::end();
}

#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input, float* output, size_t N);

int main() {
    tensor::begin("cumprod");

    size_t N = 1024;

    tensor::Buffer<float> input(N);
    tensor::Buffer<float> output(N);

    input.fill_random();

    BENCHMARK(solution(input, output, N));

    output.preview("output");

    tensor::end();
}

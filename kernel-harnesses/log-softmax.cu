#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input, float* output, size_t M, size_t N);

int main() {
    tensor::begin("log-softmax");

    size_t M = 64;
    size_t N = 64;

    tensor::Buffer<float> input(M * N);
    tensor::Buffer<float> output(M * N);

    input.fill_random();

    BENCHMARK(solution(input, output, M, N));

    output.preview("output");

    tensor::end();
}

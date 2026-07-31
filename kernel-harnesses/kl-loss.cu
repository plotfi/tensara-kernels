#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n);

int main() {
    tensor::begin("kl-loss");

    size_t n = 1024;

    tensor::Buffer<float> predictions(n);
    tensor::Buffer<float> targets(n);
    tensor::Buffer<float> output(1);

    predictions.fill_random();
    targets.fill_random();

    BENCHMARK(solution(predictions, targets, output, n));

    output.preview("output");

    tensor::end();
}

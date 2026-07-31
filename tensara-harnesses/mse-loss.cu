#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* predictions, const float* targets, float* output, const size_t* shape, size_t ndim);

int main() {
    tensor::begin("mse-loss");

    size_t ndim = 2;

    tensor::Buffer<float> predictions(64 * 64);
    tensor::Buffer<float> targets(64 * 64);
    tensor::Buffer<float> output(1);
    tensor::Buffer<size_t> shape(ndim);

    predictions.fill_random();
    targets.fill_random();
    shape.set({64, 64});

    BENCHMARK(solution(predictions, targets, output, shape, ndim));

    output.preview("output");

    tensor::end();
}

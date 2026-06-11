#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* predictions, const float* targets, float* output, const size_t* shape, size_t ndim);

int main() {
    harness::begin("mse-loss");

    size_t ndim = 2;

    harness::Buffer<float> predictions(64 * 64);
    harness::Buffer<float> targets(64 * 64);
    harness::Buffer<float> output(1);
    harness::Buffer<size_t> shape(ndim);

    predictions.fill_random();
    targets.fill_random();
    shape.set({64, 64});

    BENCHMARK(solution(predictions, targets, output, shape, ndim));

    output.preview("output");

    harness::end();
}

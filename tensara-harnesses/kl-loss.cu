#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n);

int main() {
    harness::begin("kl-loss");

    size_t n = 1024;

    harness::Buffer<float> predictions(n);
    harness::Buffer<float> targets(n);
    harness::Buffer<float> output(1);

    predictions.fill_random();
    targets.fill_random();

    BENCHMARK(solution(predictions, targets, output, n));

    output.preview("output");

    printf("Done.\n");
    return 0;
}

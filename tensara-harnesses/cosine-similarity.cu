#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n, size_t d);

int main() {
    harness::begin("cosine-similarity");

    size_t n = 64;
    size_t d = 128;

    harness::Buffer<float> predictions(n * d);
    harness::Buffer<float> targets(n * d);
    harness::Buffer<float> output(n);

    predictions.fill_random();
    targets.fill_random();

    BENCHMARK(solution(predictions, targets, output, n, d));

    output.preview("output");

    printf("Done.\n");
    return 0;
}

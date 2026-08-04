#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n);

int main() {
    tensor::begin("huber-loss");

    // 16M elements: large enough that the kernel is memory-bandwidth-bound, so
    // BENCHMARK reflects the kernel rather than fixed launch overhead. (The
    // correctness test in tests/loss-reduce.cu uses its own small n.)
    size_t n = tensor::bench_size("N", 1 << 24);

    tensor::Buffer<float> predictions(n);
    tensor::Buffer<float> targets(n);
    tensor::Buffer<float> output(1);

    predictions.fill_random();
    targets.fill_random();

    BENCHMARK(solution(predictions, targets, output, n));

    output.preview("output");

    tensor::end();
}

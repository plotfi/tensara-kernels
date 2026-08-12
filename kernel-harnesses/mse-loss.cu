#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* predictions, const float* targets, float* output, const size_t* shape, size_t ndim);

int main() {
    tensor::begin("mse-loss");

    size_t ndim = 2;
    size_t M = tensor::bench_size("M", 64);   // rows
    size_t N = tensor::bench_size("N", 64);   // cols

    tensor::Buffer<float> predictions(M * N);
    tensor::Buffer<float> targets(M * N);
    tensor::Buffer<float> output(1);
    tensor::Buffer<size_t> shape(ndim);

    predictions.fill_random();
    targets.fill_random();
    shape.set({M, N});

    BENCHMARK(solution(predictions, targets, output, shape, ndim));

    output.preview("output");

    tensor::end();
}

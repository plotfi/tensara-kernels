#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* image, int num_bins, float* histogram, size_t height, size_t width);

int main() {
    tensor::begin("histogram");

    int num_bins = tensor::bench_size("NUM_BINS", 256);
    size_t height = tensor::bench_size("HEIGHT", 64);
    size_t width = tensor::bench_size("WIDTH", 64);

    tensor::Buffer<float> image(height * width);
    tensor::Buffer<float> histogram(num_bins);

    image.fill_random();

    BENCHMARK(solution(image, num_bins, histogram, height, width));

    histogram.preview("histogram");

    tensor::end();
}

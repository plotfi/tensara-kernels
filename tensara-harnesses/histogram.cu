#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* image, int num_bins, float* histogram, size_t height, size_t width);

int main() {
    harness::begin("histogram");

    int num_bins = 256;
    size_t height = 64;
    size_t width = 64;

    harness::Buffer<float> image(height * width);
    harness::Buffer<float> histogram(num_bins);

    image.fill_random();

    BENCHMARK(solution(image, num_bins, histogram, height, width));

    histogram.preview("histogram");

    harness::end();
}

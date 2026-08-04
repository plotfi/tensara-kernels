#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* rgb_image, float* grayscale_output, size_t height, size_t width, size_t channels);

int main() {
    tensor::begin("grayscale");

    size_t height = tensor::bench_size("HEIGHT", 64);
    size_t width = tensor::bench_size("WIDTH", 64);
    size_t channels = tensor::bench_size("CHANNELS", 3);

    tensor::Buffer<float> rgb_image(height * width * channels);
    tensor::Buffer<float> grayscale_output(height * width);

    rgb_image.fill_random();

    BENCHMARK(solution(rgb_image, grayscale_output, height, width, channels));

    grayscale_output.preview("grayscale_output");

    tensor::end();
}

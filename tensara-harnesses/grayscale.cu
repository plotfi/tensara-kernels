#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* rgb_image, float* grayscale_output, size_t height, size_t width, size_t channels);

int main() {
    harness::begin("grayscale");

    size_t height = 64;
    size_t width = 64;
    size_t channels = 3;

    harness::Buffer<float> rgb_image(height * width * channels);
    harness::Buffer<float> grayscale_output(height * width);

    rgb_image.fill_random();

    BENCHMARK(solution(rgb_image, grayscale_output, height, width, channels));

    grayscale_output.preview("grayscale_output");

    harness::end();
}

#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input_image, int kernel_size, float* output_image, size_t height, size_t width);

int main() {
    harness::begin("box-blur");

    int kernel_size = 3;
    size_t height = 64;
    size_t width = 64;

    harness::Buffer<float> input_image(height * width);
    harness::Buffer<float> output_image(height * width);

    input_image.fill_random();

    BENCHMARK(solution(input_image, kernel_size, output_image, height, width));

    output_image.preview("output_image");

    harness::end();
}

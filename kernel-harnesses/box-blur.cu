#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input_image, int kernel_size, float* output_image, size_t height, size_t width);

int main() {
    tensor::begin("box-blur");

    int kernel_size = 3;
    size_t height = tensor::bench_size("HEIGHT", 64);
    size_t width = tensor::bench_size("WIDTH", 64);

    tensor::Buffer<float> input_image(height * width);
    tensor::Buffer<float> output_image(height * width);

    input_image.fill_random();

    BENCHMARK(solution(input_image, kernel_size, output_image, height, width));

    output_image.preview("output_image");

    tensor::end();
}

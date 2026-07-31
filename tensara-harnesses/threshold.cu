#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input_image, float threshold_value, float* output_image, size_t height, size_t width);

int main() {
    tensor::begin("threshold");

    float threshold_value = 0.5f;
    size_t height = 64;
    size_t width = 64;

    tensor::Buffer<float> input_image(height * width);
    tensor::Buffer<float> output_image(height * width);

    input_image.fill_random();

    BENCHMARK(solution(input_image, threshold_value, output_image, height, width));

    output_image.preview("output_image");

    tensor::end();
}

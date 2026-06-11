#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input_image, float* output_image, size_t height, size_t width);

int main() {
    harness::begin("edge-detect");

    size_t height = 64;
    size_t width = 64;

    harness::Buffer<float> input_image(height * width);
    harness::Buffer<float> output_image(height * width);

    input_image.fill_random();

    BENCHMARK(solution(input_image, output_image, height, width));

    output_image.preview("output_image");

    printf("Done.\n");
    return 0;
}

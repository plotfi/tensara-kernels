#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* image, const float* kernel, float* output, size_t H, size_t W, size_t Kh, size_t Kw);

int main() {
    harness::begin("conv2d-relu-hardswish");

    size_t H = 64;
    size_t W = 64;
    size_t Kh = 3;
    size_t Kw = 3;

    harness::Buffer<float> image(H * W);
    harness::Buffer<float> kernel(Kh * Kw);
    harness::Buffer<float> output(H * W);

    image.fill_random();
    kernel.fill_random();

    BENCHMARK(solution(image, kernel, output, H, W, Kh, Kw));

    output.preview("output");

    harness::end();
}

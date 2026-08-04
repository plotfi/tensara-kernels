#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* image, const float* kernel, float* output, size_t H, size_t W, size_t Kh, size_t Kw);

int main() {
    tensor::begin("conv2d-relu-hardswish");

    size_t H = tensor::bench_size("H", 64);
    size_t W = tensor::bench_size("W", 64);
    size_t Kh = 3;
    size_t Kw = 3;

    tensor::Buffer<float> image(H * W);
    tensor::Buffer<float> kernel(Kh * Kw);
    tensor::Buffer<float> output(H * W);

    image.fill_random();
    kernel.fill_random();

    BENCHMARK(solution(image, kernel, output, H, W, Kh, Kw));

    output.preview("output");

    tensor::end();
}

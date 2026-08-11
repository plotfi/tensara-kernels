#include "../tensor-lib/tensor.cuh"

// CS4803 Lab 3 — 2D convolution (shared-constant), 5x5 filter, size x size image.
extern "C" void solution(const float* filter, const float* N, float* P, size_t size);

int main() {
    tensor::begin("cs4803dgc-lab3-conv2d-shared-constant");

    size_t size = 512;   // multiple of 16 (block tiling)

    tensor::Buffer<float> filter(5 * 5);
    tensor::Buffer<float> N(size * size);
    tensor::Buffer<float> P(size * size);

    filter.fill_random();
    N.fill_random();

    BENCHMARK(solution(filter, N, P, size));

    P.preview("P");

    tensor::end();
}

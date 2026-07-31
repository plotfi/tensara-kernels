#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input, int dim, float* output, const size_t* shape, size_t ndim);

int main() {
    tensor::begin("mean-dim");

    int dim = 1;
    size_t ndim = 2;

    tensor::Buffer<float> input(64 * 64);
    tensor::Buffer<float> output(64);
    tensor::Buffer<size_t> shape(ndim);

    input.fill_random();
    shape.set({64, 64});

    BENCHMARK(solution(input, dim, output, shape, ndim));

    output.preview("output");

    tensor::end();
}

#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input, int dim, int* output, const int* shape, int ndim);

int main() {
    tensor::begin("argmax");

    int dim = 1;
    int ndim = 2;

    tensor::Buffer<float> input(64 * 64);
    tensor::Buffer<int> output(64);
    tensor::Buffer<int> shape(ndim);

    input.fill_random();
    shape.set({64, 64});

    BENCHMARK(solution(input, dim, output, shape, ndim));

    output.preview("output");

    tensor::end();
}

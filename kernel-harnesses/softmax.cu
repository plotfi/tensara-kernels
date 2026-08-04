#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input, int dim, float* output, const size_t* shape, size_t ndim);

int main() {
    tensor::begin("softmax");

    int dim = 1;
    size_t ndim = 2;
    size_t M = tensor::bench_size("M", 64);   // rows
    size_t N = tensor::bench_size("N", 64);   // reduced dim (dim=1)

    tensor::Buffer<float> input(M * N);
    tensor::Buffer<float> output(M * N);
    tensor::Buffer<size_t> shape(ndim);

    input.fill_random();
    shape.set({M, N});

    BENCHMARK(solution(input, dim, output, shape, ndim));

    output.preview("output");

    tensor::end();
}

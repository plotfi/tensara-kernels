#include "../tensor-lib/tensor.cuh"

// Fused conv1d -> maxpool1d. `weight` is the conv kernel (length K); the pool
// params (kernel_size/stride/padding/dilation) act on the conv output.
extern "C" void solution(const float* input, const float* weight, float* output,
                         size_t N, size_t K,
                         int kernel_size, int stride, int padding, int dilation);

int main() {
    tensor::begin("conv1d-maxpool1d");

    size_t N = 1024;   // input length
    size_t K = 5;      // conv kernel size
    int kernel_size = 3, stride = 2, padding = 1, dilation = 1;   // pool params

    int Lout = (static_cast<int>(N) + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    tensor::Buffer<float> input(N);
    tensor::Buffer<float> weight(K);
    tensor::Buffer<float> output(Lout);

    input.fill_random();
    weight.fill_random();

    BENCHMARK(solution(input, weight, output, N, K, kernel_size, stride, padding, dilation));

    output.preview("output");

    tensor::end();
}

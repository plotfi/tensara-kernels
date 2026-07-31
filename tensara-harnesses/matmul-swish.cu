#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input_matrix, const float* weight_matrix, const float* bias, float scaling_factor, float* output, size_t batch_size, size_t in_features, size_t out_features);

int main() {
    tensor::begin("matmul-swish");

    float scaling_factor = 1.0f;
    size_t batch_size = 8;
    size_t in_features = 64;
    size_t out_features = 32;

    tensor::Buffer<float> input_matrix(batch_size * in_features);
    tensor::Buffer<float> weight_matrix(out_features * in_features);
    tensor::Buffer<float> bias(out_features);
    tensor::Buffer<float> output(batch_size * out_features);

    input_matrix.fill_random();
    weight_matrix.fill_random();
    bias.fill_random();

    BENCHMARK(solution(input_matrix, weight_matrix, bias, scaling_factor, output, batch_size, in_features, out_features));

    output.preview("output");

    tensor::end();
}

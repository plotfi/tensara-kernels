#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void solution(const float* input_matrix, const float* weight_matrix, const float* bias, float scaling_factor, float* output, size_t batch_size, size_t in_features, size_t out_features);

int main(int argc, char** argv) {
    printf("=== matmul-swish ===\n");
    srand(42);

    float scaling_factor = 1.0f;
    size_t batch_size = 8;
    size_t in_features = 64;
    size_t out_features = 32;

    size_t input_matrix_count = batch_size * in_features;
    size_t weight_matrix_count = out_features * in_features;
    size_t bias_count = out_features;
    size_t output_count = batch_size * out_features;

    float* h_input_matrix = new float[input_matrix_count];
    float* h_weight_matrix = new float[weight_matrix_count];
    float* h_bias = new float[bias_count];
    float* h_output = new float[output_count];

    for (size_t i = 0; i < input_matrix_count; i++)
        h_input_matrix[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    for (size_t i = 0; i < weight_matrix_count; i++)
        h_weight_matrix[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    for (size_t i = 0; i < bias_count; i++)
        h_bias[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_output, 0, output_count * sizeof(float));

    float* d_input_matrix;
    cudaMalloc(&d_input_matrix, input_matrix_count * sizeof(float));
    float* d_weight_matrix;
    cudaMalloc(&d_weight_matrix, weight_matrix_count * sizeof(float));
    float* d_bias;
    cudaMalloc(&d_bias, bias_count * sizeof(float));
    float* d_output;
    cudaMalloc(&d_output, output_count * sizeof(float));

    cudaMemcpy(d_input_matrix, h_input_matrix, input_matrix_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_matrix, h_weight_matrix, weight_matrix_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, bias_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, output_count * sizeof(float));

    for (int _w = 0; _w < 3; _w++)
        solution(d_input_matrix, d_weight_matrix, d_bias, scaling_factor, d_output, batch_size, in_features, out_features);
    cudaDeviceSynchronize();

    cudaEvent_t _perf_start, _perf_stop;
    cudaEventCreate(&_perf_start);
    cudaEventCreate(&_perf_stop);
    const int _perf_iters = 100;
    cudaEventRecord(_perf_start);
    for (int _i = 0; _i < _perf_iters; _i++)
        solution(d_input_matrix, d_weight_matrix, d_bias, scaling_factor, d_output, batch_size, in_features, out_features);
    cudaEventRecord(_perf_stop);
    cudaEventSynchronize(_perf_stop);
    float _perf_ms = 0.0f;
    cudaEventElapsedTime(&_perf_ms, _perf_start, _perf_stop);
    cudaEventDestroy(_perf_start);
    cudaEventDestroy(_perf_stop);
    printf("Avg kernel time: %.4f ms (over %d iters)\n", _perf_ms / _perf_iters, _perf_iters);

    cudaMemcpy(h_output, d_output, output_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output output (first 10): ");
    for (size_t i = 0; i < 10 && i < output_count; i++)
        printf("%f ", h_output[i]);
    printf("\n");

    cudaFree(d_input_matrix);
    cudaFree(d_weight_matrix);
    cudaFree(d_bias);
    cudaFree(d_output);
    delete[] h_input_matrix;
    delete[] h_weight_matrix;
    delete[] h_bias;
    delete[] h_output;

    printf("Done.\n");
    return 0;
}

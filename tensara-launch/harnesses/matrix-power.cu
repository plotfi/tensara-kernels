#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void solution(const float* input_matrix, size_t n, float* output_matrix, size_t size);

int main(int argc, char** argv) {
    printf("=== matrix-power ===\n");
    srand(42);

    size_t n = 3;
    size_t size = 8;

    size_t input_matrix_count = size * size;
    size_t output_matrix_count = size * size;

    float* h_input_matrix = new float[input_matrix_count];
    float* h_output_matrix = new float[output_matrix_count];

    for (size_t i = 0; i < input_matrix_count; i++)
        h_input_matrix[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_output_matrix, 0, output_matrix_count * sizeof(float));

    float* d_input_matrix;
    cudaMalloc(&d_input_matrix, input_matrix_count * sizeof(float));
    float* d_output_matrix;
    cudaMalloc(&d_output_matrix, output_matrix_count * sizeof(float));

    cudaMemcpy(d_input_matrix, h_input_matrix, input_matrix_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output_matrix, 0, output_matrix_count * sizeof(float));

    for (int _w = 0; _w < 3; _w++)
        solution(d_input_matrix, n, d_output_matrix, size);
    cudaDeviceSynchronize();

    cudaEvent_t _perf_start, _perf_stop;
    cudaEventCreate(&_perf_start);
    cudaEventCreate(&_perf_stop);
    const int _perf_iters = 100;
    cudaEventRecord(_perf_start);
    for (int _i = 0; _i < _perf_iters; _i++)
        solution(d_input_matrix, n, d_output_matrix, size);
    cudaEventRecord(_perf_stop);
    cudaEventSynchronize(_perf_stop);
    float _perf_ms = 0.0f;
    cudaEventElapsedTime(&_perf_ms, _perf_start, _perf_stop);
    cudaEventDestroy(_perf_start);
    cudaEventDestroy(_perf_stop);
    printf("Avg kernel time: %.4f ms (over %d iters)\n", _perf_ms / _perf_iters, _perf_iters);

    cudaMemcpy(h_output_matrix, d_output_matrix, output_matrix_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output output_matrix (first 10): ");
    for (size_t i = 0; i < 10 && i < output_matrix_count; i++)
        printf("%f ", h_output_matrix[i]);
    printf("\n");

    cudaFree(d_input_matrix);
    cudaFree(d_output_matrix);
    delete[] h_input_matrix;
    delete[] h_output_matrix;

    printf("Done.\n");
    return 0;
}

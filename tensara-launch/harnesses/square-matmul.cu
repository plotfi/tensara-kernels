#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t n);

int main(int argc, char** argv) {
    printf("=== square-matmul ===\n");
    srand(42);

    size_t n = 64;

    size_t input_a_count = n * n;
    size_t input_b_count = n * n;
    size_t output_c_count = n * n;

    float* h_input_a = new float[input_a_count];
    float* h_input_b = new float[input_b_count];
    float* h_output_c = new float[output_c_count];

    for (size_t i = 0; i < input_a_count; i++)
        h_input_a[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    for (size_t i = 0; i < input_b_count; i++)
        h_input_b[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_output_c, 0, output_c_count * sizeof(float));

    float* d_input_a;
    cudaMalloc(&d_input_a, input_a_count * sizeof(float));
    float* d_input_b;
    cudaMalloc(&d_input_b, input_b_count * sizeof(float));
    float* d_output_c;
    cudaMalloc(&d_output_c, output_c_count * sizeof(float));

    cudaMemcpy(d_input_a, h_input_a, input_a_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_b, h_input_b, input_b_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output_c, 0, output_c_count * sizeof(float));

    for (int _w = 0; _w < 3; _w++)
        solution(d_input_a, d_input_b, d_output_c, n);
    cudaDeviceSynchronize();

    cudaEvent_t _perf_start, _perf_stop;
    cudaEventCreate(&_perf_start);
    cudaEventCreate(&_perf_stop);
    const int _perf_iters = 100;
    cudaEventRecord(_perf_start);
    for (int _i = 0; _i < _perf_iters; _i++)
        solution(d_input_a, d_input_b, d_output_c, n);
    cudaEventRecord(_perf_stop);
    cudaEventSynchronize(_perf_stop);
    float _perf_ms = 0.0f;
    cudaEventElapsedTime(&_perf_ms, _perf_start, _perf_stop);
    cudaEventDestroy(_perf_start);
    cudaEventDestroy(_perf_stop);
    printf("Avg kernel time: %.4f ms (over %d iters)\n", _perf_ms / _perf_iters, _perf_iters);

    cudaMemcpy(h_output_c, d_output_c, output_c_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output output_c (first 10): ");
    for (size_t i = 0; i < 10 && i < output_c_count; i++)
        printf("%f ", h_output_c[i]);
    printf("\n");

    cudaFree(d_input_a);
    cudaFree(d_input_b);
    cudaFree(d_output_c);
    delete[] h_input_a;
    delete[] h_input_b;
    delete[] h_output_c;

    printf("Done.\n");
    return 0;
}

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H);

int main(int argc, char** argv) {
    printf("=== avg-pool-1d ===\n");
    srand(42);

    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
    size_t H = 1024;

    size_t input_count = H;
    size_t output_count = H;

    float* h_input = new float[input_count];
    float* h_output = new float[output_count];

    for (size_t i = 0; i < input_count; i++)
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_output, 0, output_count * sizeof(float));

    float* d_input;
    cudaMalloc(&d_input, input_count * sizeof(float));
    float* d_output;
    cudaMalloc(&d_output, output_count * sizeof(float));

    cudaMemcpy(d_input, h_input, input_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, output_count * sizeof(float));

    for (int _w = 0; _w < 3; _w++)
        solution(d_input, kernel_size, stride, padding, d_output, H);
    cudaDeviceSynchronize();

    cudaEvent_t _perf_start, _perf_stop;
    cudaEventCreate(&_perf_start);
    cudaEventCreate(&_perf_stop);
    const int _perf_iters = 100;
    cudaEventRecord(_perf_start);
    for (int _i = 0; _i < _perf_iters; _i++)
        solution(d_input, kernel_size, stride, padding, d_output, H);
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

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;

    printf("Done.\n");
    return 0;
}

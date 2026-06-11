#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void solution(const float* A, float* min_weight, size_t n);

int main(int argc, char** argv) {
    printf("=== min-spanning-tree ===\n");
    srand(42);

    size_t n = 64;

    size_t A_count = n * n;
    size_t min_weight_count = 1;

    float* h_A = new float[A_count];
    float* h_min_weight = new float[min_weight_count];

    for (size_t i = 0; i < A_count; i++)
        h_A[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_min_weight, 0, min_weight_count * sizeof(float));

    float* d_A;
    cudaMalloc(&d_A, A_count * sizeof(float));
    float* d_min_weight;
    cudaMalloc(&d_min_weight, min_weight_count * sizeof(float));

    cudaMemcpy(d_A, h_A, A_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_min_weight, 0, min_weight_count * sizeof(float));

    for (int _w = 0; _w < 3; _w++)
        solution(d_A, d_min_weight, n);
    cudaDeviceSynchronize();

    cudaEvent_t _perf_start, _perf_stop;
    cudaEventCreate(&_perf_start);
    cudaEventCreate(&_perf_stop);
    const int _perf_iters = 100;
    cudaEventRecord(_perf_start);
    for (int _i = 0; _i < _perf_iters; _i++)
        solution(d_A, d_min_weight, n);
    cudaEventRecord(_perf_stop);
    cudaEventSynchronize(_perf_stop);
    float _perf_ms = 0.0f;
    cudaEventElapsedTime(&_perf_ms, _perf_start, _perf_stop);
    cudaEventDestroy(_perf_start);
    cudaEventDestroy(_perf_stop);
    printf("Avg kernel time: %.4f ms (over %d iters)\n", _perf_ms / _perf_iters, _perf_iters);

    cudaMemcpy(h_min_weight, d_min_weight, min_weight_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output min_weight (first 10): ");
    for (size_t i = 0; i < 10 && i < min_weight_count; i++)
        printf("%f ", h_min_weight[i]);
    printf("\n");

    cudaFree(d_A);
    cudaFree(d_min_weight);
    delete[] h_A;
    delete[] h_min_weight;

    printf("Done.\n");
    return 0;
}

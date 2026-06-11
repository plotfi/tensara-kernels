#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void solution(const int* a, int* b, size_t n);

int main(int argc, char** argv) {
    printf("=== array-sort ===\n");
    srand(42);

    size_t n = 1024;

    size_t a_count = n;
    size_t b_count = n;

    int* h_a = new int[a_count];
    int* h_b = new int[b_count];

    for (size_t i = 0; i < a_count; i++)
        h_a[i] = rand() % 201 - 100;
    memset(h_b, 0, b_count * sizeof(int));

    int* d_a;
    cudaMalloc(&d_a, a_count * sizeof(int));
    int* d_b;
    cudaMalloc(&d_b, b_count * sizeof(int));

    cudaMemcpy(d_a, h_a, a_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_b, 0, b_count * sizeof(int));

    for (int _w = 0; _w < 3; _w++)
        solution(d_a, d_b, n);
    cudaDeviceSynchronize();

    cudaEvent_t _perf_start, _perf_stop;
    cudaEventCreate(&_perf_start);
    cudaEventCreate(&_perf_stop);
    const int _perf_iters = 100;
    cudaEventRecord(_perf_start);
    for (int _i = 0; _i < _perf_iters; _i++)
        solution(d_a, d_b, n);
    cudaEventRecord(_perf_stop);
    cudaEventSynchronize(_perf_stop);
    float _perf_ms = 0.0f;
    cudaEventElapsedTime(&_perf_ms, _perf_start, _perf_stop);
    cudaEventDestroy(_perf_start);
    cudaEventDestroy(_perf_stop);
    printf("Avg kernel time: %.4f ms (over %d iters)\n", _perf_ms / _perf_iters, _perf_iters);

    cudaMemcpy(h_b, d_b, b_count * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Output b (first 10): ");
    for (size_t i = 0; i < 10 && i < b_count; i++)
        printf("%d ", h_b[i]);
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    delete[] h_a;
    delete[] h_b;

    printf("Done.\n");
    return 0;
}

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void solution(const uint8_t* q_a, const uint8_t* scale_a, const uint8_t* q_b, const uint8_t* scale_b, float* c, size_t m, size_t n, size_t k);

int main(int argc, char** argv) {
    printf("=== mxfp8-gemm ===\n");
    srand(42);

    size_t m = 64;
    size_t n = 64;
    size_t k = 64;

    size_t q_a_count = m * k;
    size_t scale_a_count = m * (k / 32);
    size_t q_b_count = k * n;
    size_t scale_b_count = k * (n / 32);
    size_t c_count = m * n;

    uint8_t* h_q_a = new uint8_t[q_a_count];
    uint8_t* h_scale_a = new uint8_t[scale_a_count];
    uint8_t* h_q_b = new uint8_t[q_b_count];
    uint8_t* h_scale_b = new uint8_t[scale_b_count];
    float* h_c = new float[c_count];

    for (size_t i = 0; i < q_a_count; i++)
        h_q_a[i] = static_cast<uint8_t>(rand() % 256);
    for (size_t i = 0; i < scale_a_count; i++)
        h_scale_a[i] = static_cast<uint8_t>(rand() % 256);
    for (size_t i = 0; i < q_b_count; i++)
        h_q_b[i] = static_cast<uint8_t>(rand() % 256);
    for (size_t i = 0; i < scale_b_count; i++)
        h_scale_b[i] = static_cast<uint8_t>(rand() % 256);
    memset(h_c, 0, c_count * sizeof(float));

    uint8_t* d_q_a;
    cudaMalloc(&d_q_a, q_a_count * sizeof(uint8_t));
    uint8_t* d_scale_a;
    cudaMalloc(&d_scale_a, scale_a_count * sizeof(uint8_t));
    uint8_t* d_q_b;
    cudaMalloc(&d_q_b, q_b_count * sizeof(uint8_t));
    uint8_t* d_scale_b;
    cudaMalloc(&d_scale_b, scale_b_count * sizeof(uint8_t));
    float* d_c;
    cudaMalloc(&d_c, c_count * sizeof(float));

    cudaMemcpy(d_q_a, h_q_a, q_a_count * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_a, h_scale_a, scale_a_count * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_b, h_q_b, q_b_count * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_b, h_scale_b, scale_b_count * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, c_count * sizeof(float));

    for (int _w = 0; _w < 3; _w++)
        solution(d_q_a, d_scale_a, d_q_b, d_scale_b, d_c, m, n, k);
    cudaDeviceSynchronize();

    cudaEvent_t _perf_start, _perf_stop;
    cudaEventCreate(&_perf_start);
    cudaEventCreate(&_perf_stop);
    const int _perf_iters = 100;
    cudaEventRecord(_perf_start);
    for (int _i = 0; _i < _perf_iters; _i++)
        solution(d_q_a, d_scale_a, d_q_b, d_scale_b, d_c, m, n, k);
    cudaEventRecord(_perf_stop);
    cudaEventSynchronize(_perf_stop);
    float _perf_ms = 0.0f;
    cudaEventElapsedTime(&_perf_ms, _perf_start, _perf_stop);
    cudaEventDestroy(_perf_start);
    cudaEventDestroy(_perf_stop);
    printf("Avg kernel time: %.4f ms (over %d iters)\n", _perf_ms / _perf_iters, _perf_iters);

    cudaMemcpy(h_c, d_c, c_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output c (first 10): ");
    for (size_t i = 0; i < 10 && i < c_count; i++)
        printf("%f ", h_c[i]);
    printf("\n");

    cudaFree(d_q_a);
    cudaFree(d_scale_a);
    cudaFree(d_q_b);
    cudaFree(d_scale_b);
    cudaFree(d_c);
    delete[] h_q_a;
    delete[] h_scale_a;
    delete[] h_q_b;
    delete[] h_scale_b;
    delete[] h_c;

    printf("Done.\n");
    return 0;
}

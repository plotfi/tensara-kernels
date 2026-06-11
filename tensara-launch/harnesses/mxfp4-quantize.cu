#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void solution(const float* a, uint8_t* q, uint8_t* scale, size_t m, size_t k);

int main(int argc, char** argv) {
    printf("=== mxfp4-quantize ===\n");
    srand(42);

    size_t m = 64;
    size_t k = 64;

    size_t a_count = m * k;
    size_t q_count = m * k / 2;
    size_t scale_count = m * (k / 32);

    float* h_a = new float[a_count];
    uint8_t* h_q = new uint8_t[q_count];
    uint8_t* h_scale = new uint8_t[scale_count];

    for (size_t i = 0; i < a_count; i++)
        h_a[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_q, 0, q_count * sizeof(uint8_t));
    memset(h_scale, 0, scale_count * sizeof(uint8_t));

    float* d_a;
    cudaMalloc(&d_a, a_count * sizeof(float));
    uint8_t* d_q;
    cudaMalloc(&d_q, q_count * sizeof(uint8_t));
    uint8_t* d_scale;
    cudaMalloc(&d_scale, scale_count * sizeof(uint8_t));

    cudaMemcpy(d_a, h_a, a_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_q, 0, q_count * sizeof(uint8_t));
    cudaMemset(d_scale, 0, scale_count * sizeof(uint8_t));

    for (int _w = 0; _w < 3; _w++)
        solution(d_a, d_q, d_scale, m, k);
    cudaDeviceSynchronize();

    cudaEvent_t _perf_start, _perf_stop;
    cudaEventCreate(&_perf_start);
    cudaEventCreate(&_perf_stop);
    const int _perf_iters = 100;
    cudaEventRecord(_perf_start);
    for (int _i = 0; _i < _perf_iters; _i++)
        solution(d_a, d_q, d_scale, m, k);
    cudaEventRecord(_perf_stop);
    cudaEventSynchronize(_perf_stop);
    float _perf_ms = 0.0f;
    cudaEventElapsedTime(&_perf_ms, _perf_start, _perf_stop);
    cudaEventDestroy(_perf_start);
    cudaEventDestroy(_perf_stop);
    printf("Avg kernel time: %.4f ms (over %d iters)\n", _perf_ms / _perf_iters, _perf_iters);

    cudaMemcpy(h_q, d_q, q_count * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_scale, d_scale, scale_count * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    printf("Output q (first 10): ");
    for (size_t i = 0; i < 10 && i < q_count; i++)
        printf("%u ", h_q[i]);
    printf("\n");
    printf("Output scale (first 10): ");
    for (size_t i = 0; i < 10 && i < scale_count; i++)
        printf("%u ", h_scale[i]);
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_q);
    cudaFree(d_scale);
    delete[] h_a;
    delete[] h_q;
    delete[] h_scale;

    printf("Done.\n");
    return 0;
}

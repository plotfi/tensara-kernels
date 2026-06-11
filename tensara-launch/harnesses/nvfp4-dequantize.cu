#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

extern "C" void solution(const uint8_t* q, const __nv_fp8_e4m3* scale, float sf_g, float* out, size_t m, size_t k);

int main(int argc, char** argv) {
    printf("=== nvfp4-dequantize ===\n");
    srand(42);

    float sf_g = 1.0f;
    size_t m = 64;
    size_t k = 64;

    size_t q_count = m * k / 2;
    size_t scale_count = m * (k / 16);
    size_t out_count = m * k;

    uint8_t* h_q = new uint8_t[q_count];
    __nv_fp8_e4m3* h_scale = new __nv_fp8_e4m3[scale_count];
    float* h_out = new float[out_count];

    for (size_t i = 0; i < q_count; i++)
        h_q[i] = static_cast<uint8_t>(rand() % 256);
    for (size_t i = 0; i < scale_count; i++)
        h_scale[i] = static_cast<__nv_fp8_e4m3>(0);
    memset(h_out, 0, out_count * sizeof(float));

    uint8_t* d_q;
    cudaMalloc(&d_q, q_count * sizeof(uint8_t));
    __nv_fp8_e4m3* d_scale;
    cudaMalloc(&d_scale, scale_count * sizeof(__nv_fp8_e4m3));
    float* d_out;
    cudaMalloc(&d_out, out_count * sizeof(float));

    cudaMemcpy(d_q, h_q, q_count * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, h_scale, scale_count * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, out_count * sizeof(float));

    for (int _w = 0; _w < 3; _w++)
        solution(d_q, d_scale, sf_g, d_out, m, k);
    cudaDeviceSynchronize();

    cudaEvent_t _perf_start, _perf_stop;
    cudaEventCreate(&_perf_start);
    cudaEventCreate(&_perf_stop);
    const int _perf_iters = 100;
    cudaEventRecord(_perf_start);
    for (int _i = 0; _i < _perf_iters; _i++)
        solution(d_q, d_scale, sf_g, d_out, m, k);
    cudaEventRecord(_perf_stop);
    cudaEventSynchronize(_perf_stop);
    float _perf_ms = 0.0f;
    cudaEventElapsedTime(&_perf_ms, _perf_start, _perf_stop);
    cudaEventDestroy(_perf_start);
    cudaEventDestroy(_perf_stop);
    printf("Avg kernel time: %.4f ms (over %d iters)\n", _perf_ms / _perf_iters, _perf_iters);

    cudaMemcpy(h_out, d_out, out_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output out (first 10): ");
    for (size_t i = 0; i < 10 && i < out_count; i++)
        printf("%f ", h_out[i]);
    printf("\n");

    cudaFree(d_q);
    cudaFree(d_scale);
    cudaFree(d_out);
    delete[] h_q;
    delete[] h_scale;
    delete[] h_out;

    printf("Done.\n");
    return 0;
}

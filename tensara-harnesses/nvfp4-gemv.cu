#include "../kernel-implementation/harness.cuh"
#include <cuda_fp16.h>
#include <cuda_fp8.h>

extern "C" void solution(const uint8_t* q_a, const __nv_fp8_e4m3* scale_a, float sf_g_a, const uint8_t* q_x, const __nv_fp8_e4m3* scale_x, float sf_g_x, half* y, size_t m, size_t k);

int main(int argc, char** argv) {
    printf("=== nvfp4-gemv ===\n");
    srand(42);

    float sf_g_a = 1.0f;
    float sf_g_x = 1.0f;
    size_t m = 64;
    size_t k = 64;

    size_t q_a_count = m * k / 2;
    size_t scale_a_count = m * (k / 16);
    size_t q_x_count = k / 2;
    size_t scale_x_count = k / 16;
    size_t y_count = m;

    uint8_t* h_q_a = new uint8_t[q_a_count];
    __nv_fp8_e4m3* h_scale_a = new __nv_fp8_e4m3[scale_a_count];
    uint8_t* h_q_x = new uint8_t[q_x_count];
    __nv_fp8_e4m3* h_scale_x = new __nv_fp8_e4m3[scale_x_count];
    half* h_y = new half[y_count];

    for (size_t i = 0; i < q_a_count; i++)
        h_q_a[i] = static_cast<uint8_t>(rand() % 256);
    for (size_t i = 0; i < scale_a_count; i++)
        h_scale_a[i] = static_cast<__nv_fp8_e4m3>(0);
    for (size_t i = 0; i < q_x_count; i++)
        h_q_x[i] = static_cast<uint8_t>(rand() % 256);
    for (size_t i = 0; i < scale_x_count; i++)
        h_scale_x[i] = static_cast<__nv_fp8_e4m3>(0);
    memset(h_y, 0, y_count * sizeof(half));

    uint8_t* d_q_a;
    cudaMalloc(&d_q_a, q_a_count * sizeof(uint8_t));
    __nv_fp8_e4m3* d_scale_a;
    cudaMalloc(&d_scale_a, scale_a_count * sizeof(__nv_fp8_e4m3));
    uint8_t* d_q_x;
    cudaMalloc(&d_q_x, q_x_count * sizeof(uint8_t));
    __nv_fp8_e4m3* d_scale_x;
    cudaMalloc(&d_scale_x, scale_x_count * sizeof(__nv_fp8_e4m3));
    half* d_y;
    cudaMalloc(&d_y, y_count * sizeof(half));

    cudaMemcpy(d_q_a, h_q_a, q_a_count * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_a, h_scale_a, scale_a_count * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_x, h_q_x, q_x_count * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_x, h_scale_x, scale_x_count * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, y_count * sizeof(half));

    BENCHMARK(solution(d_q_a, d_scale_a, sf_g_a, d_q_x, d_scale_x, sf_g_x, d_y, m, k));

    cudaMemcpy(h_y, d_y, y_count * sizeof(half), cudaMemcpyDeviceToHost);

    printf("Output y (first 10): ");
    for (size_t i = 0; i < 10 && i < y_count; i++)
        printf("%f ", __half2float(h_y[i]));
    printf("\n");

    cudaFree(d_q_a);
    cudaFree(d_scale_a);
    cudaFree(d_q_x);
    cudaFree(d_scale_x);
    cudaFree(d_y);
    delete[] h_q_a;
    delete[] h_scale_a;
    delete[] h_q_x;
    delete[] h_scale_x;
    delete[] h_y;

    printf("Done.\n");
    return 0;
}

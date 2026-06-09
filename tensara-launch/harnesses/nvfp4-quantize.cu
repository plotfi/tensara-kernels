#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

extern "C" void solution(const half* a, float sf_g, uint8_t* q, __nv_fp8_e4m3* scale, size_t m, size_t k);

int main(int argc, char** argv) {
    printf("=== nvfp4-quantize ===\n");
    srand(42);

    float sf_g = 1.0f;
    size_t m = 64;
    size_t k = 64;

    size_t a_count = m * k;
    size_t q_count = m * k / 2;
    size_t scale_count = m * (k / 16);

    half* h_a = new half[a_count];
    uint8_t* h_q = new uint8_t[q_count];
    __nv_fp8_e4m3* h_scale = new __nv_fp8_e4m3[scale_count];

    for (size_t i = 0; i < a_count; i++)
        h_a[i] = __float2half(static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f);
    memset(h_q, 0, q_count * sizeof(uint8_t));
    memset(h_scale, 0, scale_count * sizeof(__nv_fp8_e4m3));

    half* d_a;
    cudaMalloc(&d_a, a_count * sizeof(half));
    uint8_t* d_q;
    cudaMalloc(&d_q, q_count * sizeof(uint8_t));
    __nv_fp8_e4m3* d_scale;
    cudaMalloc(&d_scale, scale_count * sizeof(__nv_fp8_e4m3));

    cudaMemcpy(d_a, h_a, a_count * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_q, 0, q_count * sizeof(uint8_t));
    cudaMemset(d_scale, 0, scale_count * sizeof(__nv_fp8_e4m3));

    solution(d_a, sf_g, d_q, d_scale, m, k);
    cudaDeviceSynchronize();

    cudaMemcpy(h_q, d_q, q_count * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_scale, d_scale, scale_count * sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToHost);

    printf("Output q (first 10): ");
    for (size_t i = 0; i < 10 && i < q_count; i++)
        printf("%u ", h_q[i]);
    printf("\n");
    printf("Output scale (first 10): ");
    for (size_t i = 0; i < 10 && i < scale_count; i++)
        printf("%u ", static_cast<unsigned>(reinterpret_cast<uint8_t*>(h_scale)[i]));
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

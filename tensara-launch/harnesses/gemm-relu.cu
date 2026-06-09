#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void solution(const float* A, const float* W, const float* b, float* C, size_t B, size_t N, size_t M);

int main(int argc, char** argv) {
    printf("=== gemm-relu ===\n");
    srand(42);

    size_t B = 8;
    size_t N = 64;
    size_t M = 32;

    size_t A_count = B * N;
    size_t W_count = M * N;
    size_t b_count = M;
    size_t C_count = B * M;

    float* h_A = new float[A_count];
    float* h_W = new float[W_count];
    float* h_b = new float[b_count];
    float* h_C = new float[C_count];

    for (size_t i = 0; i < A_count; i++)
        h_A[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    for (size_t i = 0; i < W_count; i++)
        h_W[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    for (size_t i = 0; i < b_count; i++)
        h_b[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_C, 0, C_count * sizeof(float));

    float* d_A;
    cudaMalloc(&d_A, A_count * sizeof(float));
    float* d_W;
    cudaMalloc(&d_W, W_count * sizeof(float));
    float* d_b;
    cudaMalloc(&d_b, b_count * sizeof(float));
    float* d_C;
    cudaMalloc(&d_C, C_count * sizeof(float));

    cudaMemcpy(d_A, h_A, A_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, W_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, b_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, C_count * sizeof(float));

    solution(d_A, d_W, d_b, d_C, B, N, M);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, C_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output C (first 10): ");
    for (size_t i = 0; i < 10 && i < C_count; i++)
        printf("%f ", h_C[i]);
    printf("\n");

    cudaFree(d_A);
    cudaFree(d_W);
    cudaFree(d_b);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_W;
    delete[] h_b;
    delete[] h_C;

    printf("Done.\n");
    return 0;
}

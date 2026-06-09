#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void solution(const float* A, const float* B, const float* C, float alpha, float* output, size_t M, size_t N, size_t K);

int main(int argc, char** argv) {
    printf("=== gemm-multiply-leakyrelu ===\n");
    srand(42);

    float alpha = 0.01f;
    size_t M = 64;
    size_t N = 64;
    size_t K = 64;

    size_t A_count = M * K;
    size_t B_count = K * N;
    size_t C_count = M * N;
    size_t output_count = M * N;

    float* h_A = new float[A_count];
    float* h_B = new float[B_count];
    float* h_C = new float[C_count];
    float* h_output = new float[output_count];

    for (size_t i = 0; i < A_count; i++)
        h_A[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    for (size_t i = 0; i < B_count; i++)
        h_B[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    for (size_t i = 0; i < C_count; i++)
        h_C[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_output, 0, output_count * sizeof(float));

    float* d_A;
    cudaMalloc(&d_A, A_count * sizeof(float));
    float* d_B;
    cudaMalloc(&d_B, B_count * sizeof(float));
    float* d_C;
    cudaMalloc(&d_C, C_count * sizeof(float));
    float* d_output;
    cudaMalloc(&d_output, output_count * sizeof(float));

    cudaMemcpy(d_A, h_A, A_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, B_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, C_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, output_count * sizeof(float));

    solution(d_A, d_B, d_C, alpha, d_output, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, output_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output output (first 10): ");
    for (size_t i = 0; i < 10 && i < output_count; i++)
        printf("%f ", h_output[i]);
    printf("\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_output);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_output;

    printf("Done.\n");
    return 0;
}

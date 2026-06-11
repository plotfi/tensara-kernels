#include "../kernel-implementation/harness.cuh"

#include "../kernel-implementation/conv-1d.cu"
extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K);

int main(int argc, char** argv) {
    printf("=== conv-1d ===\n");
    srand(42);

    size_t N = 1024;
    size_t K = 5;

    size_t A_count = N;
    size_t B_count = K;
    size_t C_count = N;

    float* h_A = new float[A_count];
    float* h_B = new float[B_count];
    float* h_C = new float[C_count];

    for (size_t i = 0; i < A_count; i++)
        h_A[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    for (size_t i = 0; i < B_count; i++)
        h_B[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_C, 0, C_count * sizeof(float));

    float* d_A;
    cudaMalloc(&d_A, A_count * sizeof(float));
    float* d_B;
    cudaMalloc(&d_B, B_count * sizeof(float));
    float* d_C;
    cudaMalloc(&d_C, C_count * sizeof(float));

    cudaMemcpy(d_A, h_A, A_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, B_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, C_count * sizeof(float));

    BENCHMARK(solution(d_A, d_B, d_C, N, K));

    cudaMemcpy(h_C, d_C, C_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output C (first 10): ");
    for (size_t i = 0; i < 10 && i < C_count; i++)
        printf("%f ", h_C[i]);
    printf("\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    printf("Done.\n");
    return 0;
}

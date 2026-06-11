#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* A, const float* B, float* C, size_t b, size_t i, size_t j, size_t l, size_t k);

int main(int argc, char** argv) {
    printf("=== matmul-4d ===\n");
    srand(42);

    size_t b = 2;
    size_t i = 4;
    size_t j = 32;
    size_t l = 32;
    size_t k = 16;

    size_t A_count = b * i * j * k;
    size_t B_count = k * l;
    size_t C_count = b * i * j * l;

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

    BENCHMARK(solution(d_A, d_B, d_C, b, i, j, l, k));

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

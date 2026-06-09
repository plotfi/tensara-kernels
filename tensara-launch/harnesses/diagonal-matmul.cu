#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void solution(const float* diagonal_a, const float* input_b, float* output_c, size_t n, size_t m);

int main(int argc, char** argv) {
    printf("=== diagonal-matmul ===\n");
    srand(42);

    size_t n = 64;
    size_t m = 64;

    size_t diagonal_a_count = n;
    size_t input_b_count = n * m;
    size_t output_c_count = n * m;

    float* h_diagonal_a = new float[diagonal_a_count];
    float* h_input_b = new float[input_b_count];
    float* h_output_c = new float[output_c_count];

    for (size_t i = 0; i < diagonal_a_count; i++)
        h_diagonal_a[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    for (size_t i = 0; i < input_b_count; i++)
        h_input_b[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_output_c, 0, output_c_count * sizeof(float));

    float* d_diagonal_a;
    cudaMalloc(&d_diagonal_a, diagonal_a_count * sizeof(float));
    float* d_input_b;
    cudaMalloc(&d_input_b, input_b_count * sizeof(float));
    float* d_output_c;
    cudaMalloc(&d_output_c, output_c_count * sizeof(float));

    cudaMemcpy(d_diagonal_a, h_diagonal_a, diagonal_a_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_b, h_input_b, input_b_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output_c, 0, output_c_count * sizeof(float));

    solution(d_diagonal_a, d_input_b, d_output_c, n, m);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output_c, d_output_c, output_c_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output output_c (first 10): ");
    for (size_t i = 0; i < 10 && i < output_c_count; i++)
        printf("%f ", h_output_c[i]);
    printf("\n");

    cudaFree(d_diagonal_a);
    cudaFree(d_input_b);
    cudaFree(d_output_c);
    delete[] h_diagonal_a;
    delete[] h_input_b;
    delete[] h_output_c;

    printf("Done.\n");
    return 0;
}

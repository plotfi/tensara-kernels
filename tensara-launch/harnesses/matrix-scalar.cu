#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void solution(const float* input_matrix, float scalar, float* output_matrix, size_t n);

int main(int argc, char** argv) {
    printf("=== matrix-scalar ===\n");
    srand(42);

    float scalar = 2.5f;
    size_t n = 64;

    size_t input_matrix_count = n * n;
    size_t output_matrix_count = n * n;

    float* h_input_matrix = new float[input_matrix_count];
    float* h_output_matrix = new float[output_matrix_count];

    for (size_t i = 0; i < input_matrix_count; i++)
        h_input_matrix[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_output_matrix, 0, output_matrix_count * sizeof(float));

    float* d_input_matrix;
    cudaMalloc(&d_input_matrix, input_matrix_count * sizeof(float));
    float* d_output_matrix;
    cudaMalloc(&d_output_matrix, output_matrix_count * sizeof(float));

    cudaMemcpy(d_input_matrix, h_input_matrix, input_matrix_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output_matrix, 0, output_matrix_count * sizeof(float));

    solution(d_input_matrix, scalar, d_output_matrix, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output_matrix, d_output_matrix, output_matrix_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output output_matrix (first 10): ");
    for (size_t i = 0; i < 10 && i < output_matrix_count; i++)
        printf("%f ", h_output_matrix[i]);
    printf("\n");

    cudaFree(d_input_matrix);
    cudaFree(d_output_matrix);
    delete[] h_input_matrix;
    delete[] h_output_matrix;

    printf("Done.\n");
    return 0;
}

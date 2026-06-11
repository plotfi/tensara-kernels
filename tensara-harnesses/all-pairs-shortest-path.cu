#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* adj_matrix, float* output, size_t n);

int main(int argc, char** argv) {
    printf("=== all-pairs-shortest-path ===\n");
    srand(42);

    size_t n = 64;

    size_t adj_matrix_count = n * n;
    size_t output_count = n * n;

    float* h_adj_matrix = new float[adj_matrix_count];
    float* h_output = new float[output_count];

    for (size_t i = 0; i < adj_matrix_count; i++)
        h_adj_matrix[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_output, 0, output_count * sizeof(float));

    float* d_adj_matrix;
    cudaMalloc(&d_adj_matrix, adj_matrix_count * sizeof(float));
    float* d_output;
    cudaMalloc(&d_output, output_count * sizeof(float));

    cudaMemcpy(d_adj_matrix, h_adj_matrix, adj_matrix_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, output_count * sizeof(float));

    BENCHMARK(solution(d_adj_matrix, d_output, n));

    cudaMemcpy(h_output, d_output, output_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output output (first 10): ");
    for (size_t i = 0; i < 10 && i < output_count; i++)
        printf("%f ", h_output[i]);
    printf("\n");

    cudaFree(d_adj_matrix);
    cudaFree(d_output);
    delete[] h_adj_matrix;
    delete[] h_output;

    printf("Done.\n");
    return 0;
}

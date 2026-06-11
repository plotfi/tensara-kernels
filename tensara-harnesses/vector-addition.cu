#include "../kernel-implementation/harness.cuh"

#include "../kernel-implementation/vector-addition.cu"
extern "C" void solution(const float* d_input1, const float* d_input2, float* d_output, size_t n);

int main(int argc, char** argv) {
    printf("=== vector-addition ===\n");
    srand(42);

    size_t n = 1024;

    size_t d_input1_count = n;
    size_t d_input2_count = n;
    size_t d_output_count = n;

    float* h_d_input1 = new float[d_input1_count];
    float* h_d_input2 = new float[d_input2_count];
    float* h_d_output = new float[d_output_count];

    for (size_t i = 0; i < d_input1_count; i++)
        h_d_input1[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    for (size_t i = 0; i < d_input2_count; i++)
        h_d_input2[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_d_output, 0, d_output_count * sizeof(float));

    float* d_d_input1;
    cudaMalloc(&d_d_input1, d_input1_count * sizeof(float));
    float* d_d_input2;
    cudaMalloc(&d_d_input2, d_input2_count * sizeof(float));
    float* d_d_output;
    cudaMalloc(&d_d_output, d_output_count * sizeof(float));

    cudaMemcpy(d_d_input1, h_d_input1, d_input1_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d_input2, h_d_input2, d_input2_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_d_output, 0, d_output_count * sizeof(float));

    BENCHMARK(solution(d_d_input1, d_d_input2, d_d_output, n));

    cudaMemcpy(h_d_output, d_d_output, d_output_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output d_output (first 10): ");
    for (size_t i = 0; i < 10 && i < d_output_count; i++)
        printf("%f ", h_d_output[i]);
    printf("\n");

    cudaFree(d_d_input1);
    cudaFree(d_d_input2);
    cudaFree(d_d_output);
    delete[] h_d_input1;
    delete[] h_d_input2;
    delete[] h_d_output;

    printf("Done.\n");
    return 0;
}

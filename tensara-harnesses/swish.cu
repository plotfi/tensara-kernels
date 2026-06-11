#include "../kernel-implementation/harness.cuh"

#include "../kernel-implementation/activations.cu"
extern "C" void solution(const float* input, float* output, size_t n, size_t m);

int main(int argc, char** argv) {
    printf("=== swish ===\n");
    srand(42);

    size_t n = 64;
    size_t m = 64;

    size_t input_count = n * m;
    size_t output_count = n * m;

    float* h_input = new float[input_count];
    float* h_output = new float[output_count];

    for (size_t i = 0; i < input_count; i++)
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_output, 0, output_count * sizeof(float));

    float* d_input;
    cudaMalloc(&d_input, input_count * sizeof(float));
    float* d_output;
    cudaMalloc(&d_output, output_count * sizeof(float));

    cudaMemcpy(d_input, h_input, input_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, output_count * sizeof(float));

    BENCHMARK(solution(d_input, d_output, n, m));

    cudaMemcpy(h_output, d_output, output_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output output (first 10): ");
    for (size_t i = 0; i < 10 && i < output_count; i++)
        printf("%f ", h_output[i]);
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;

    printf("Done.\n");
    return 0;
}

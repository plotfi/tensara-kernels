#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input, int dim, int* output, const int* shape, int ndim);

int main(int argc, char** argv) {
    printf("=== argmin ===\n");
    srand(42);

    int dim = 1;
    int ndim = 2;

    // Default shape: 2D tensor 64x64
    int h_shape[] = {64, 64};
    ndim = 2;
    dim = 1;

    size_t input_count = 64 * 64;
    size_t output_count = 64;
    size_t shape_count = ndim;

    float* h_input = new float[input_count];
    int* h_output = new int[output_count];

    for (size_t i = 0; i < input_count; i++)
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_output, 0, output_count * sizeof(int));

    float* d_input;
    cudaMalloc(&d_input, input_count * sizeof(float));
    int* d_output;
    cudaMalloc(&d_output, output_count * sizeof(int));
    int* d_shape;
    cudaMalloc(&d_shape, ndim * sizeof(int));
    cudaMemcpy(d_shape, h_shape, ndim * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_input, h_input, input_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, output_count * sizeof(int));

    BENCHMARK(solution(d_input, dim, d_output, d_shape, ndim));

    cudaMemcpy(h_output, d_output, output_count * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Output output (first 10): ");
    for (size_t i = 0; i < 10 && i < output_count; i++)
        printf("%d ", h_output[i]);
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_shape);
    delete[] h_input;
    delete[] h_output;

    printf("Done.\n");
    return 0;
}

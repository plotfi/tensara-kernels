#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input, int dim, float* output, const size_t* shape, size_t ndim);

int main(int argc, char** argv) {
    printf("=== product-dim ===\n");
    srand(42);

    int dim = 1;
    size_t ndim = 2;

    // Default shape: 2D tensor 64x64
    size_t h_shape[] = {64, 64};
    ndim = 2;
    dim = 1;

    size_t input_count = 64 * 64;
    size_t output_count = 64;
    size_t shape_count = ndim;

    float* h_input = new float[input_count];
    float* h_output = new float[output_count];

    for (size_t i = 0; i < input_count; i++)
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_output, 0, output_count * sizeof(float));

    float* d_input;
    cudaMalloc(&d_input, input_count * sizeof(float));
    float* d_output;
    cudaMalloc(&d_output, output_count * sizeof(float));
    size_t* d_shape;
    cudaMalloc(&d_shape, ndim * sizeof(size_t));
    cudaMemcpy(d_shape, h_shape, ndim * sizeof(size_t), cudaMemcpyHostToDevice);

    cudaMemcpy(d_input, h_input, input_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, output_count * sizeof(float));

    BENCHMARK(solution(d_input, dim, d_output, d_shape, ndim));

    cudaMemcpy(h_output, d_output, output_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output output (first 10): ");
    for (size_t i = 0; i < 10 && i < output_count; i++)
        printf("%f ", h_output[i]);
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_shape);
    delete[] h_input;
    delete[] h_output;

    printf("Done.\n");
    return 0;
}

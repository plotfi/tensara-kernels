#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* image, const float* kernel, float* output, size_t H, size_t W, size_t Kh, size_t Kw);

int main(int argc, char** argv) {
    printf("=== conv2d-relu-hardswish ===\n");
    srand(42);

    size_t H = 64;
    size_t W = 64;
    size_t Kh = 3;
    size_t Kw = 3;

    size_t image_count = H * W;
    size_t kernel_count = Kh * Kw;
    size_t output_count = H * W;

    float* h_image = new float[image_count];
    float* h_kernel = new float[kernel_count];
    float* h_output = new float[output_count];

    for (size_t i = 0; i < image_count; i++)
        h_image[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    for (size_t i = 0; i < kernel_count; i++)
        h_kernel[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_output, 0, output_count * sizeof(float));

    float* d_image;
    cudaMalloc(&d_image, image_count * sizeof(float));
    float* d_kernel;
    cudaMalloc(&d_kernel, kernel_count * sizeof(float));
    float* d_output;
    cudaMalloc(&d_output, output_count * sizeof(float));

    cudaMemcpy(d_image, h_image, image_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, output_count * sizeof(float));

    BENCHMARK(solution(d_image, d_kernel, d_output, H, W, Kh, Kw));

    cudaMemcpy(h_output, d_output, output_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output output (first 10): ");
    for (size_t i = 0; i < 10 && i < output_count; i++)
        printf("%f ", h_output[i]);
    printf("\n");

    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_output);
    delete[] h_image;
    delete[] h_kernel;
    delete[] h_output;

    printf("Done.\n");
    return 0;
}

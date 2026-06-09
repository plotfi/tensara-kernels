#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void solution(const float* input_image, float threshold_value, float* output_image, size_t height, size_t width);

int main(int argc, char** argv) {
    printf("=== threshold ===\n");
    srand(42);

    float threshold_value = 0.5f;
    size_t height = 64;
    size_t width = 64;

    size_t input_image_count = height * width;
    size_t output_image_count = height * width;

    float* h_input_image = new float[input_image_count];
    float* h_output_image = new float[output_image_count];

    for (size_t i = 0; i < input_image_count; i++)
        h_input_image[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_output_image, 0, output_image_count * sizeof(float));

    float* d_input_image;
    cudaMalloc(&d_input_image, input_image_count * sizeof(float));
    float* d_output_image;
    cudaMalloc(&d_output_image, output_image_count * sizeof(float));

    cudaMemcpy(d_input_image, h_input_image, input_image_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output_image, 0, output_image_count * sizeof(float));

    solution(d_input_image, threshold_value, d_output_image, height, width);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output_image, d_output_image, output_image_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output output_image (first 10): ");
    for (size_t i = 0; i < 10 && i < output_image_count; i++)
        printf("%f ", h_output_image[i]);
    printf("\n");

    cudaFree(d_input_image);
    cudaFree(d_output_image);
    delete[] h_input_image;
    delete[] h_output_image;

    printf("Done.\n");
    return 0;
}

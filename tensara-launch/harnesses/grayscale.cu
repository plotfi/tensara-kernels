#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void solution(const float* rgb_image, float* grayscale_output, size_t height, size_t width, size_t channels);

int main(int argc, char** argv) {
    printf("=== grayscale ===\n");
    srand(42);

    size_t height = 64;
    size_t width = 64;
    size_t channels = 3;

    size_t rgb_image_count = height * width * channels;
    size_t grayscale_output_count = height * width;

    float* h_rgb_image = new float[rgb_image_count];
    float* h_grayscale_output = new float[grayscale_output_count];

    for (size_t i = 0; i < rgb_image_count; i++)
        h_rgb_image[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_grayscale_output, 0, grayscale_output_count * sizeof(float));

    float* d_rgb_image;
    cudaMalloc(&d_rgb_image, rgb_image_count * sizeof(float));
    float* d_grayscale_output;
    cudaMalloc(&d_grayscale_output, grayscale_output_count * sizeof(float));

    cudaMemcpy(d_rgb_image, h_rgb_image, rgb_image_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_grayscale_output, 0, grayscale_output_count * sizeof(float));

    solution(d_rgb_image, d_grayscale_output, height, width, channels);
    cudaDeviceSynchronize();

    cudaMemcpy(h_grayscale_output, d_grayscale_output, grayscale_output_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output grayscale_output (first 10): ");
    for (size_t i = 0; i < 10 && i < grayscale_output_count; i++)
        printf("%f ", h_grayscale_output[i]);
    printf("\n");

    cudaFree(d_rgb_image);
    cudaFree(d_grayscale_output);
    delete[] h_rgb_image;
    delete[] h_grayscale_output;

    printf("Done.\n");
    return 0;
}

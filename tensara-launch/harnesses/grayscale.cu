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

    for (int _w = 0; _w < 3; _w++)
        solution(d_rgb_image, d_grayscale_output, height, width, channels);
    cudaDeviceSynchronize();

    cudaEvent_t _perf_start, _perf_stop;
    cudaEventCreate(&_perf_start);
    cudaEventCreate(&_perf_stop);
    const int _perf_iters = 100;
    cudaEventRecord(_perf_start);
    for (int _i = 0; _i < _perf_iters; _i++)
        solution(d_rgb_image, d_grayscale_output, height, width, channels);
    cudaEventRecord(_perf_stop);
    cudaEventSynchronize(_perf_stop);
    float _perf_ms = 0.0f;
    cudaEventElapsedTime(&_perf_ms, _perf_start, _perf_stop);
    cudaEventDestroy(_perf_start);
    cudaEventDestroy(_perf_stop);
    printf("Avg kernel time: %.4f ms (over %d iters)\n", _perf_ms / _perf_iters, _perf_iters);

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

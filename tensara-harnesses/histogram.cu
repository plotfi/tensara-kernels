#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void solution(const float* image, int num_bins, float* histogram, size_t height, size_t width);

int main(int argc, char** argv) {
    printf("=== histogram ===\n");
    srand(42);

    int num_bins = 256;
    size_t height = 64;
    size_t width = 64;

    size_t image_count = height * width;
    size_t histogram_count = num_bins;

    float* h_image = new float[image_count];
    float* h_histogram = new float[histogram_count];

    for (size_t i = 0; i < image_count; i++)
        h_image[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_histogram, 0, histogram_count * sizeof(float));

    float* d_image;
    cudaMalloc(&d_image, image_count * sizeof(float));
    float* d_histogram;
    cudaMalloc(&d_histogram, histogram_count * sizeof(float));

    cudaMemcpy(d_image, h_image, image_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, histogram_count * sizeof(float));

    for (int _w = 0; _w < 3; _w++)
        solution(d_image, num_bins, d_histogram, height, width);
    cudaDeviceSynchronize();

    cudaEvent_t _perf_start, _perf_stop;
    cudaEventCreate(&_perf_start);
    cudaEventCreate(&_perf_stop);
    const int _perf_iters = 100;
    cudaEventRecord(_perf_start);
    for (int _i = 0; _i < _perf_iters; _i++)
        solution(d_image, num_bins, d_histogram, height, width);
    cudaEventRecord(_perf_stop);
    cudaEventSynchronize(_perf_stop);
    float _perf_ms = 0.0f;
    cudaEventElapsedTime(&_perf_ms, _perf_start, _perf_stop);
    cudaEventDestroy(_perf_start);
    cudaEventDestroy(_perf_stop);
    printf("Avg kernel time: %.4f ms (over %d iters)\n", _perf_ms / _perf_iters, _perf_iters);

    cudaMemcpy(h_histogram, d_histogram, histogram_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output histogram (first 10): ");
    for (size_t i = 0; i < 10 && i < histogram_count; i++)
        printf("%f ", h_histogram[i]);
    printf("\n");

    cudaFree(d_image);
    cudaFree(d_histogram);
    delete[] h_image;
    delete[] h_histogram;

    printf("Done.\n");
    return 0;
}

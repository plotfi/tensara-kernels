#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void solution(const float* X, float* Y, size_t B, size_t D);

int main(int argc, char** argv) {
    printf("=== l2-norm ===\n");
    srand(42);

    size_t B = 8;
    size_t D = 64;

    size_t X_count = B * D;
    size_t Y_count = B * D;

    float* h_X = new float[X_count];
    float* h_Y = new float[Y_count];

    for (size_t i = 0; i < X_count; i++)
        h_X[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_Y, 0, Y_count * sizeof(float));

    float* d_X;
    cudaMalloc(&d_X, X_count * sizeof(float));
    float* d_Y;
    cudaMalloc(&d_Y, Y_count * sizeof(float));

    cudaMemcpy(d_X, h_X, X_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_Y, 0, Y_count * sizeof(float));

    for (int _w = 0; _w < 3; _w++)
        solution(d_X, d_Y, B, D);
    cudaDeviceSynchronize();

    cudaEvent_t _perf_start, _perf_stop;
    cudaEventCreate(&_perf_start);
    cudaEventCreate(&_perf_stop);
    const int _perf_iters = 100;
    cudaEventRecord(_perf_start);
    for (int _i = 0; _i < _perf_iters; _i++)
        solution(d_X, d_Y, B, D);
    cudaEventRecord(_perf_stop);
    cudaEventSynchronize(_perf_stop);
    float _perf_ms = 0.0f;
    cudaEventElapsedTime(&_perf_ms, _perf_start, _perf_stop);
    cudaEventDestroy(_perf_start);
    cudaEventDestroy(_perf_stop);
    printf("Avg kernel time: %.4f ms (over %d iters)\n", _perf_ms / _perf_iters, _perf_iters);

    cudaMemcpy(h_Y, d_Y, Y_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output Y (first 10): ");
    for (size_t i = 0; i < 10 && i < Y_count; i++)
        printf("%f ", h_Y[i]);
    printf("\n");

    cudaFree(d_X);
    cudaFree(d_Y);
    delete[] h_X;
    delete[] h_Y;

    printf("Done.\n");
    return 0;
}

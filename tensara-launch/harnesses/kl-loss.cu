#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n);

int main(int argc, char** argv) {
    printf("=== kl-loss ===\n");
    srand(42);

    size_t n = 1024;

    size_t predictions_count = n;
    size_t targets_count = n;
    size_t output_count = 1;

    float* h_predictions = new float[predictions_count];
    float* h_targets = new float[targets_count];
    float* h_output = new float[output_count];

    for (size_t i = 0; i < predictions_count; i++)
        h_predictions[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    for (size_t i = 0; i < targets_count; i++)
        h_targets[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_output, 0, output_count * sizeof(float));

    float* d_predictions;
    cudaMalloc(&d_predictions, predictions_count * sizeof(float));
    float* d_targets;
    cudaMalloc(&d_targets, targets_count * sizeof(float));
    float* d_output;
    cudaMalloc(&d_output, output_count * sizeof(float));

    cudaMemcpy(d_predictions, h_predictions, predictions_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets, targets_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, output_count * sizeof(float));

    for (int _w = 0; _w < 3; _w++)
        solution(d_predictions, d_targets, d_output, n);
    cudaDeviceSynchronize();

    cudaEvent_t _perf_start, _perf_stop;
    cudaEventCreate(&_perf_start);
    cudaEventCreate(&_perf_stop);
    const int _perf_iters = 100;
    cudaEventRecord(_perf_start);
    for (int _i = 0; _i < _perf_iters; _i++)
        solution(d_predictions, d_targets, d_output, n);
    cudaEventRecord(_perf_stop);
    cudaEventSynchronize(_perf_stop);
    float _perf_ms = 0.0f;
    cudaEventElapsedTime(&_perf_ms, _perf_start, _perf_stop);
    cudaEventDestroy(_perf_start);
    cudaEventDestroy(_perf_stop);
    printf("Avg kernel time: %.4f ms (over %d iters)\n", _perf_ms / _perf_iters, _perf_iters);

    cudaMemcpy(h_output, d_output, output_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output output (first 10): ");
    for (size_t i = 0; i < 10 && i < output_count; i++)
        printf("%f ", h_output[i]);
    printf("\n");

    cudaFree(d_predictions);
    cudaFree(d_targets);
    cudaFree(d_output);
    delete[] h_predictions;
    delete[] h_targets;
    delete[] h_output;

    printf("Done.\n");
    return 0;
}

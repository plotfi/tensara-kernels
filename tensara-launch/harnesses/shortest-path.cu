#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void solution(const float* d_adj_matrix, int source, float* d_distances, size_t n);

int main(int argc, char** argv) {
    printf("=== shortest-path ===\n");
    srand(42);

    int source = 0;
    size_t n = 64;

    size_t d_adj_matrix_count = n * n;
    size_t d_distances_count = n;

    float* h_d_adj_matrix = new float[d_adj_matrix_count];
    float* h_d_distances = new float[d_distances_count];

    for (size_t i = 0; i < d_adj_matrix_count; i++)
        h_d_adj_matrix[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_d_distances, 0, d_distances_count * sizeof(float));

    float* d_d_adj_matrix;
    cudaMalloc(&d_d_adj_matrix, d_adj_matrix_count * sizeof(float));
    float* d_d_distances;
    cudaMalloc(&d_d_distances, d_distances_count * sizeof(float));

    cudaMemcpy(d_d_adj_matrix, h_d_adj_matrix, d_adj_matrix_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_d_distances, 0, d_distances_count * sizeof(float));

    for (int _w = 0; _w < 3; _w++)
        solution(d_d_adj_matrix, source, d_d_distances, n);
    cudaDeviceSynchronize();

    cudaEvent_t _perf_start, _perf_stop;
    cudaEventCreate(&_perf_start);
    cudaEventCreate(&_perf_stop);
    const int _perf_iters = 100;
    cudaEventRecord(_perf_start);
    for (int _i = 0; _i < _perf_iters; _i++)
        solution(d_d_adj_matrix, source, d_d_distances, n);
    cudaEventRecord(_perf_stop);
    cudaEventSynchronize(_perf_stop);
    float _perf_ms = 0.0f;
    cudaEventElapsedTime(&_perf_ms, _perf_start, _perf_stop);
    cudaEventDestroy(_perf_start);
    cudaEventDestroy(_perf_stop);
    printf("Avg kernel time: %.4f ms (over %d iters)\n", _perf_ms / _perf_iters, _perf_iters);

    cudaMemcpy(h_d_distances, d_d_distances, d_distances_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output d_distances (first 10): ");
    for (size_t i = 0; i < 10 && i < d_distances_count; i++)
        printf("%f ", h_d_distances[i]);
    printf("\n");

    cudaFree(d_d_adj_matrix);
    cudaFree(d_d_distances);
    delete[] h_d_adj_matrix;
    delete[] h_d_distances;

    printf("Done.\n");
    return 0;
}

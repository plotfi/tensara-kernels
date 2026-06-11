#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void solution(const uint64_t* xs, const uint64_t* ys, uint64_t p, uint64_t* out_xy, size_t n);

int main(int argc, char** argv) {
    printf("=== ecc-point-negation ===\n");
    srand(42);

    uint64_t p = 18446744073709551557ULL;
    size_t n = 1024;

    size_t xs_count = n;
    size_t ys_count = n;
    size_t out_xy_count = n * 2;

    uint64_t* h_xs = new uint64_t[xs_count];
    uint64_t* h_ys = new uint64_t[ys_count];
    uint64_t* h_out_xy = new uint64_t[out_xy_count];

    for (size_t i = 0; i < xs_count; i++)
        h_xs[i] = static_cast<uint64_t>(rand());
    for (size_t i = 0; i < ys_count; i++)
        h_ys[i] = static_cast<uint64_t>(rand());
    memset(h_out_xy, 0, out_xy_count * sizeof(uint64_t));

    uint64_t* d_xs;
    cudaMalloc(&d_xs, xs_count * sizeof(uint64_t));
    uint64_t* d_ys;
    cudaMalloc(&d_ys, ys_count * sizeof(uint64_t));
    uint64_t* d_out_xy;
    cudaMalloc(&d_out_xy, out_xy_count * sizeof(uint64_t));

    cudaMemcpy(d_xs, h_xs, xs_count * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ys, h_ys, ys_count * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemset(d_out_xy, 0, out_xy_count * sizeof(uint64_t));

    for (int _w = 0; _w < 3; _w++)
        solution(d_xs, d_ys, p, d_out_xy, n);
    cudaDeviceSynchronize();

    cudaEvent_t _perf_start, _perf_stop;
    cudaEventCreate(&_perf_start);
    cudaEventCreate(&_perf_stop);
    const int _perf_iters = 100;
    cudaEventRecord(_perf_start);
    for (int _i = 0; _i < _perf_iters; _i++)
        solution(d_xs, d_ys, p, d_out_xy, n);
    cudaEventRecord(_perf_stop);
    cudaEventSynchronize(_perf_stop);
    float _perf_ms = 0.0f;
    cudaEventElapsedTime(&_perf_ms, _perf_start, _perf_stop);
    cudaEventDestroy(_perf_start);
    cudaEventDestroy(_perf_stop);
    printf("Avg kernel time: %.4f ms (over %d iters)\n", _perf_ms / _perf_iters, _perf_iters);

    cudaMemcpy(h_out_xy, d_out_xy, out_xy_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    printf("Output out_xy (first 10): ");
    for (size_t i = 0; i < 10 && i < out_xy_count; i++)
        printf("%llu ", h_out_xy[i]);
    printf("\n");

    cudaFree(d_xs);
    cudaFree(d_ys);
    cudaFree(d_out_xy);
    delete[] h_xs;
    delete[] h_ys;
    delete[] h_out_xy;

    printf("Done.\n");
    return 0;
}

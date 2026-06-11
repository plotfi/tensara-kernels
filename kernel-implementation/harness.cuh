#pragma once

// Shared launch/benchmark helpers for the Tensara harnesses.
//
// Every harness allocates buffers, fills inputs, calls solution(), and prints
// results. The warmup + timed-loop measurement around solution() is identical
// across all harnesses, so it lives here.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

namespace harness {

// Run `launch` `warmup` times untimed (to warm clocks/caches), then time
// `iters` launches with CUDA events and print the average per-launch time.
template <typename Fn>
inline void benchmark(Fn&& launch, int warmup = 3, int iters = 100) {
    for (int i = 0; i < warmup; i++)
        launch();
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++)
        launch();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Avg kernel time: %.4f ms (over %d iters)\n", ms / iters, iters);
}

} // namespace harness

// Benchmark a solution() invocation, e.g.:
//   BENCHMARK(solution(d_a, d_b, d_c, m, n, k));
// The call expression is captured by reference and replayed each iteration.
#define BENCHMARK(...) ::harness::benchmark([&]() { __VA_ARGS__; })

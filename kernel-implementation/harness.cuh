#pragma once

// Shared launch/benchmark helpers for the Tensara harnesses.
//
// Every harness does the same thing: seed RNG, allocate host+device buffers,
// fill inputs, run solution() under a timed loop, then copy outputs back and
// print a preview. All of that boilerplate lives here so each harness only has
// to describe its buffers and call solution().

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <initializer_list>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace harness {

// Print the harness banner and seed the RNG (fixed seed for reproducibility).
inline void begin(const char* name) {
    printf("=== %s ===\n", name);
    srand(42);
}

// Print the harness footer.
inline void end() {
    printf("Done.\n");
}

// ---- per-element random fill, dispatched by type ------------------------
inline void fill_random(float* p, size_t n) {
    for (size_t i = 0; i < n; i++)
        p[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
}
inline void fill_random(int* p, size_t n) {
    for (size_t i = 0; i < n; i++)
        p[i] = rand() % 201 - 100;
}
inline void fill_random(uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; i++)
        p[i] = static_cast<uint8_t>(rand() % 256);
}
inline void fill_random(uint32_t* p, size_t n) {
    for (size_t i = 0; i < n; i++)
        p[i] = static_cast<uint32_t>(rand());
}
inline void fill_random(uint64_t* p, size_t n) {
    for (size_t i = 0; i < n; i++)
        p[i] = static_cast<uint64_t>(rand());
}
inline void fill_random(half* p, size_t n) {
    for (size_t i = 0; i < n; i++)
        p[i] = __float2half(static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f);
}
inline void fill_random(__nv_fp8_e4m3* p, size_t n) {
    // fp8 inputs are seeded to zero in the original harnesses.
    for (size_t i = 0; i < n; i++)
        p[i] = static_cast<__nv_fp8_e4m3>(0);
}

// ---- per-element preview print, dispatched by type ----------------------
inline void preview_value(float v)        { printf("%f ", v); }
inline void preview_value(int v)          { printf("%d ", v); }
inline void preview_value(uint8_t v)      { printf("%u ", static_cast<unsigned>(v)); }
inline void preview_value(uint32_t v)     { printf("%u ", v); }
inline void preview_value(uint64_t v)     { printf("%llu ", static_cast<unsigned long long>(v)); }
inline void preview_value(half v)         { printf("%f ", __half2float(v)); }
inline void preview_value(__nv_fp8_e4m3 v){ printf("%u ", static_cast<unsigned>(*reinterpret_cast<uint8_t*>(&v))); }

// RAII host+device buffer pair. Construction allocates both and zeroes them;
// destruction frees both. Implicitly converts to the device pointer so it can
// be passed straight to solution().
template <typename T>
struct Buffer {
    T* host = nullptr;
    T* dev  = nullptr;
    size_t count = 0;

    explicit Buffer(size_t n) : count(n) {
        host = new T[n];
        cudaMalloc(&dev, n * sizeof(T));
        memset(host, 0, n * sizeof(T));
        cudaMemset(dev, 0, n * sizeof(T));
    }
    ~Buffer() { cudaFree(dev); delete[] host; }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    // Fill host with random data (by element type) and upload to device.
    Buffer& fill_random() {
        ::harness::fill_random(host, count);
        cudaMemcpy(dev, host, count * sizeof(T), cudaMemcpyHostToDevice);
        return *this;
    }

    // Set explicit host values (e.g. a shape vector) and upload to device.
    Buffer& set(std::initializer_list<T> values) {
        size_t i = 0;
        for (T v : values)
            if (i < count) host[i++] = v;
        cudaMemcpy(dev, host, count * sizeof(T), cudaMemcpyHostToDevice);
        return *this;
    }

    // Copy device -> host and print the first `k` elements.
    void preview(const char* label, size_t k = 10) {
        cudaMemcpy(host, dev, count * sizeof(T), cudaMemcpyDeviceToHost);
        printf("Output %s (first 10): ", label);
        for (size_t i = 0; i < k && i < count; i++)
            ::harness::preview_value(host[i]);
        printf("\n");
    }

    operator T*() { return dev; }
};

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
//   BENCHMARK(solution(a, b, c, m, n, k));
// The call expression is captured by reference and replayed each iteration.
#define BENCHMARK(...) ::harness::benchmark([&]() { __VA_ARGS__; })

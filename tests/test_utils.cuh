#pragma once

// Shared helpers for the Tensara correctness tests.
//
// Each test file computes a CPU reference for a kernel's `solution()` and
// compares it against the GPU output with a tolerance. A test's main() returns
// 0 on success and 1 on failure so the run_tests.sh launcher can aggregate a
// pass/fail summary from process exit codes.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace test {

// Fixed seed so failures are reproducible (independent of the benchmark RNG).
inline void seed() { srand(123); }

// Fill a host array with uniform random values in [lo, hi].
inline void fill_random(float* p, size_t n, float lo = -1.0f, float hi = 1.0f) {
    for (size_t i = 0; i < n; i++)
        p[i] = lo + (hi - lo) * (static_cast<float>(rand()) / RAND_MAX);
}

// Integer fill in [lo, hi].
inline void fill_random_int(int* p, size_t n, int lo = -100, int hi = 100) {
    for (size_t i = 0; i < n; i++)
        p[i] = lo + rand() % (hi - lo + 1);
}

// Small unsigned 64-bit fill (rand() is < 2^31, well below any 64-bit modulus).
inline void fill_random_u64(uint64_t* p, size_t n) {
    for (size_t i = 0; i < n; i++)
        p[i] = static_cast<uint64_t>(rand());
}

// Abort with a message if the last CUDA call (or kernel launch) failed.
inline void check_cuda(const char* where) {
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  CUDA error at %s: %s\n", where, cudaGetErrorString(err));
        exit(2);
    }
}

// RAII device buffer, zero-initialised on construction. Implicitly converts to
// the device pointer so it can be passed straight to solution().
template <typename T>
struct DBuf {
    T* dev = nullptr;
    size_t count = 0;

    explicit DBuf(size_t n) : count(n) {
        cudaMalloc(&dev, n * sizeof(T));
        cudaMemset(dev, 0, n * sizeof(T));
    }
    ~DBuf() { cudaFree(dev); }
    DBuf(const DBuf&) = delete;
    DBuf& operator=(const DBuf&) = delete;

    void upload(const T* h)   { cudaMemcpy(dev, h, count * sizeof(T), cudaMemcpyHostToDevice); }
    void download(T* h) const { cudaMemcpy(h, dev, count * sizeof(T), cudaMemcpyDeviceToHost); }

    operator T*() { return dev; }
};

// Count elements where |got - expected| exceeds atol + rtol*|expected|.
// Prints up to a few offending elements for debugging.
inline int compare(const char* name, const float* got, const float* expected,
                   size_t n, float rtol = 1e-3f, float atol = 1e-4f) {
    int bad = 0;
    for (size_t i = 0; i < n; i++) {
        float g = got[i], e = expected[i];
        bool g_nan = std::isnan(g), e_nan = std::isnan(e);
        float diff = fabsf(g - e);
        float tol = atol + rtol * fabsf(e);
        if (g_nan != e_nan || (!g_nan && !(diff <= tol))) {
            if (bad < 5)
                printf("  [%s] mismatch at %zu: got %g expected %g (diff %g > tol %g)\n",
                       name, i, g, e, diff, tol);
            bad++;
        }
    }
    return bad;
}

// Exact integer comparison (for argmax/argmin, array-sort, ecc, ...).
template <typename T>
inline int compare_int(const char* name, const T* got, const T* expected, size_t n) {
    int bad = 0;
    for (size_t i = 0; i < n; i++) {
        if (got[i] != expected[i]) {
            if (bad < 5)
                printf("  [%s] mismatch at %zu: got %lld expected %lld\n",
                       name, i, static_cast<long long>(got[i]),
                       static_cast<long long>(expected[i]));
            bad++;
        }
    }
    return bad;
}

// Print PASS/FAIL and return the process exit code.
inline int report(const char* name, int bad, size_t n) {
    if (bad == 0) {
        printf("PASS: %s (%zu elements)\n", name, n);
        return 0;
    }
    printf("FAIL: %s (%d / %zu mismatches)\n", name, bad, n);
    return 1;
}

} // namespace test

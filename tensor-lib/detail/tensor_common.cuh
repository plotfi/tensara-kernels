#pragma once

// Backend-neutral helpers shared by every harness backend.
//
// The parts of the harness that have nothing to do with CUDA or Metal: the
// banner, RNG seeding, and the per-element random-fill / preview routines
// dispatched by element type. The CUDA backend adds half / fp8 overloads on top
// of these; the Metal backend uses them as-is.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <initializer_list>

namespace tensor {

// Print the harness banner and seed the RNG (fixed seed for reproducibility).
inline void begin(const char* name) {
    printf("=== %s ===\n", name);
    srand(42);
}

// Print the harness footer.
inline void end() {
    printf("Done.\n");
}

// ---- env-controlled benchmark sizes -------------------------------------
// Harnesses default to small sizes, so an unscaled run mostly measures kernel-
// launch latency. Two env knobs let you scale up to hit bandwidth/throughput:
//
//   TENSOR_SCALE=<k>   multiply EVERY size dimension by k (default 1)
//   TENSOR_<DIM>=<n>   set one dimension absolutely (overrides SCALE for it),
//                      e.g. TENSOR_N, TENSOR_M, TENSOR_K, TENSOR_WIDTH, ...
//
// Values accept a plain integer or a k/m/g suffix (1024-based), e.g. 16m, 4k.
// With no env set, sizes are exactly the compiled-in defaults (zero change).
// Structural params (stride, padding, kernel_size, dilation, dim) are never
// routed through this — only tensor sizes are.
inline bool parse_size(const char* s, size_t& out) {
    if (!s || !*s) return false;
    char* endp = nullptr;
    unsigned long long v = strtoull(s, &endp, 10);
    if (endp == s) return false;
    size_t mult = 1;
    switch (*endp) {
        case 'k': case 'K': mult = 1024ull; endp++; break;
        case 'm': case 'M': mult = 1024ull * 1024; endp++; break;
        case 'g': case 'G': mult = 1024ull * 1024 * 1024; endp++; break;
        case '\0': break;
        default: return false;   // trailing garbage
    }
    if (*endp != '\0') return false;
    out = static_cast<size_t>(v) * mult;
    return true;
}

// Global multiplier from TENSOR_SCALE (default 1). Read once.
inline size_t size_scale() {
    static const size_t s = [] {
        size_t v;
        if (const char* e = std::getenv("TENSOR_SCALE"); e && parse_size(e, v) && v > 0)
            return v;
        return static_cast<size_t>(1);
    }();
    return s;
}

// A benchmark size dimension. `dim` names the env override, e.g. bench_size("N",
// 1024) is settable with TENSOR_N=... (absolute) or scaled by TENSOR_SCALE.
inline size_t bench_size(const char* dim, size_t fallback) {
    char var[64];
    snprintf(var, sizeof(var), "TENSOR_%s", dim);
    size_t v;
    if (const char* e = std::getenv(var); e && parse_size(e, v) && v > 0)
        return v;                          // absolute per-dim override
    return fallback * size_scale();          // else the default, globally scaled
}

// ---- per-element random fill, dispatched by type ------------------------
inline void fill_random(float* p, size_t n) {
    for (size_t i = 0; i < n; i++)
        p[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
}
inline void fill_random(double* p, size_t n) {
    for (size_t i = 0; i < n; i++)
        p[i] = static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0;
}
inline void fill_random(int* p, size_t n) {
    for (size_t i = 0; i < n; i++)
        p[i] = rand() % 201 - 100;
}
inline void fill_random(uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; i++)
        p[i] = static_cast<uint8_t>(rand() % 256);
}
inline void fill_random(int8_t* p, size_t n) {
    for (size_t i = 0; i < n; i++)
        p[i] = static_cast<int8_t>((rand() & 0xff) - 128);   // [-128, 127]
}
inline void fill_random(uint32_t* p, size_t n) {
    for (size_t i = 0; i < n; i++)
        p[i] = static_cast<uint32_t>(rand());
}
inline void fill_random(uint64_t* p, size_t n) {
    for (size_t i = 0; i < n; i++)
        p[i] = static_cast<uint64_t>(rand());
}

// ---- per-element preview print, dispatched by type ----------------------
inline void preview_value(float v)    { printf("%f ", v); }
inline void preview_value(double v)   { printf("%f ", v); }
inline void preview_value(int v)      { printf("%d ", v); }
inline void preview_value(uint8_t v)  { printf("%u ", static_cast<unsigned>(v)); }
inline void preview_value(int8_t v)   { printf("%d ", static_cast<int>(v)); }
inline void preview_value(uint32_t v) { printf("%u ", v); }
inline void preview_value(uint64_t v) { printf("%llu ", static_cast<unsigned long long>(v)); }

} // namespace tensor

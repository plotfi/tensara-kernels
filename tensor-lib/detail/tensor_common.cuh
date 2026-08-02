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

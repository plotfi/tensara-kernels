// Solution stub for "ecc-point-negation".
// TODO: implement the body below. The signature is derived from
//       kernel-harnesses/ecc-point-negation.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/ecc-point-negation.exe
//   ./build/bin/ecc-point-negation.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const uint64_t* xs, const uint64_t* ys, uint64_t p, uint64_t* out_xy, size_t n) {
    // TODO: implement ecc-point-negation
}

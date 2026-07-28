// Solution stub for "poly-multiply-ff".
// TODO: implement the body below. The signature is derived from
//       tensara-harnesses/poly-multiply-ff.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/poly-multiply-ff.exe
//   ./build/bin/poly-multiply-ff.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Note: all pointer arguments are device pointers.
extern "C" void solution(const uint32_t* d_input1, const uint32_t* d_input2, uint32_t* d_output, size_t n) {
    // TODO: implement poly-multiply-ff
}

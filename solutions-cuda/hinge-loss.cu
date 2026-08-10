// Solution stub for "hinge-loss".
// The signature is derived from
// kernel-harnesses/hinge-loss.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/hinge-loss.exe
//   ./build/bin/hinge-loss.exe
#include "../kernel-implementation/loss.cuh"

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float *predictions, const float *targets,
                         float *output, size_t n) {
  launch_block_reduce<HingeLossImpl, reduce_sum,
                      /*BLOCK_SIZE=*/512>(predictions, targets, output, n);
}

// Solution stub for "huber-loss".
// The signature is derived from kernel-harnesses/huber-loss.cu and must stay in
// sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/huber-loss.exe
//   ./build/bin/huber-loss.exe
#include "../kernel-implementation/loss.cuh"

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float *predictions, const float *targets,
                         float *output, size_t n) {
  launch_block_reduce<HuberLossImpl, reduce_sum,
                      /*BLOCK_SIZE=*/512>(predictions, targets, output, n);
}

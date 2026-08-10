// kl-loss: output[0] = mean_i targets_i * (log targets_i - log predictions_i).
// Inputs are strictly positive. Uses the shared loss + block-reduction framework.
#include "../kernel-implementation/loss.cuh"

// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) {
  launch_block_reduce<KLLossImpl, reduce_sum, /*BLOCK_SIZE=*/512>(predictions, targets, output, n);
}

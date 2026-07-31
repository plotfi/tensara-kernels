// Solution for "rms-norm" (per-row reduction + writeback).
#include "../kernel-implementation/reduction.cuh"

extern "C" void solution(const float* X, float* Y, size_t B, size_t N) {
    launch_reduce<RMSNormOps, 512>(X, Y, static_cast<int>(B), static_cast<int>(N));
}

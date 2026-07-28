// Solution for "max-normalize" (per-row reduction + writeback).
#include "../kernel-implementation/reduction.cuh"

extern "C" void solution(const float* X, float* Y, size_t B, size_t D) {
    launch_reduce<AbsMaxNormOps, 256>(X, Y, static_cast<int>(B), static_cast<int>(D));
}

// Solution for "mean-subtract" (per-row reduction + writeback).
#include "../kernel-implementation/reduction.cuh"

extern "C" void solution(const float* X, float* Y, size_t B, size_t D) {
    launch_reduce<MeanSubOps, 256>(X, Y, static_cast<int>(B), static_cast<int>(D));
}

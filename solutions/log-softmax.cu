// Solution for "log-softmax" (per-row reduction + writeback).
#include "../kernel-implementation/reduction.cuh"

extern "C" void solution(const float* input, float* output, size_t M, size_t N) {
    launch_reduce<LogSoftmaxOps, 512>(input, output, static_cast<int>(M), static_cast<int>(N));
}
